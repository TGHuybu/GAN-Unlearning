import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt 

from repulsion import RepulsionLoss
import torch.autograd as autograd
import torch.optim as optim

from noise import make_noise, mixing_noise
# from evaluation import fid_score


try:
    import wandb

except ImportError:
    wandb = None


from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths



def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def unlearn(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, rloss, mask=None):
    # consolidation_loss_history = []
    # iter_history = []

    loader = sample_data(loader)    

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img = next(loader)
        real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img

        fake_pred = discriminator(fake_img)
        real_pred = discriminator(real_img_aug)
        d_loss = d_logistic_loss(real_pred, fake_pred)
        # loss_dis_adapt = ewc_dis._compute_consolidation_loss(ewc_dis.weight)
        loss_dict["d"] = d_loss 
        #+ loss_dis_adapt
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug)
            #loss_dis_adapt = ewc_dis._compute_consolidation_loss(ewc_dis.weight)

            r1_loss = d_r1_loss(real_pred, real_img) 
            #+ loss_dis_adapt

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        g_repulsion_loss = rloss._compute_repulsion_loss(generator)
        g_loss += g_repulsion_loss
        loss_dict["g"] = g_loss
        generator.zero_grad()
        g_loss.backward()

        # APPLY MASK IF ANY
        if mask is not None:
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    param.grad = mask[name].to(device) * param.grad
        g_optim.step()
        
        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            g_reg_repulsion_loss = rloss._compute_repulsion_loss(generator)
            weighted_path_loss += g_reg_repulsion_loss
            weighted_path_loss.backward()
            
            # APPLY MASK IF ANY
            if mask is not None:
                for name, param in generator.named_parameters():
                    if param.grad is not None:
                        param.grad = mask[name].to(device) * param.grad
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f};"
                    f"augment: {ada_aug_p:.4f};" f"repulsion loss: {g_repulsion_loss:.4f};" 
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                        
                    }
                )

            if i % args.save_interval == 0:
                with torch.no_grad():
                    g_ema.eval()
                    sample, _ = g_ema([sample_z])
                    utils.save_image(
                        sample,
                        f"{args.outdir}/sample/{str(i).zfill(6)}.png",
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if i % args.save_interval == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p,
                    },
                    f"{args.outdir}/model/{str(i).zfill(6)}.pt",
                )
                
    torch.save(
        {
            "g": g_module.state_dict(),
            "d": d_module.state_dict(),
            "g_ema": g_ema.state_dict(),
            "g_optim": g_optim.state_dict(),
            "d_optim": d_optim.state_dict(),
            "args": args,
            "ada_aug_p": ada_aug_p,
        },
        f"{args.outdir}/model/unlearned.pt",
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 unlearner")

    parser.add_argument("--path", type=str, help="path to the lmdb dataset")
    parser.add_argument("--ckpt", type=str, help="path to the trained model")
    # parser.add_argument("--adapted_ckpt", type=str, default=None, help="path to adapted model")
    parser.add_argument(
        "--adapted_ckpt", 
        type=str, nargs="+", default=None, help="path(s) to adapted model(s)"
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )

    # unlearn args
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2exp",
        choices=["l2inv", "l2neg", "l2exp", "l2ens"],
        help="repulsion loss type [ l2inv | l2neg | l2exp | l2ens ]",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="param for 'l2ens' loss type",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.7,
        help="param for 'l2ens' loss type",
    )
    parser.add_argument(
        "--rl_weight",
        type=float,
        default=10,
        help="repulsion loss weight",
    )
    parser.add_argument(
        "--dweight",
        type=float,
        default=1,
        help="param difference weight",
    )
    parser.add_argument(
        "--exp_lambda",
        type=float,
        default=0.05,
        help="param for 'l2exp' loss type",
    )
    parser.add_argument(
        "--scale_rloss", action="store_true", help="[DO NOT USE, EFFECT NOT CERTAIN] scale repulsion between 0 and 10"
    )

    # other args
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
        help="probability update interval of the adaptive augmentation",
    )
    
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator

    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    # initialize model
    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    # not sure if unlearning D is needed, create just in case
    gs_adapted = []
    for i in range(len(args.adapted_ckpt)):
        gs_adapted.append(Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
        ).to(device))
        # d_adapted = Discriminator(
        #     args.size, channel_multiplier=args.channel_multiplier
        # ).to(device)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    # load original model
    print("load original model:", args.ckpt)
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage, weights_only=True)
    try:
        ckpt_name = os.path.basename(args.ckpt)
        args.start_iter = int(os.path.splitext(ckpt_name)[0])
    except ValueError:
        pass
    generator.load_state_dict(ckpt["g"], strict=False)
    discriminator.load_state_dict(ckpt["d"], strict=False)
    g_ema.load_state_dict(ckpt["g_ema"], strict=False)
    g_optim.load_state_dict(ckpt["g_optim"])
    d_optim.load_state_dict(ckpt["d_optim"])

    # load adapted model (only need the weights)
    for i, ackpt in enumerate(args.adapted_ckpt):
        print("load adapted model:", ackpt)
        adapted_ckpt = torch.load(ackpt, map_location=lambda storage, loc: storage, weights_only=False)
        gs_adapted[i].load_state_dict(adapted_ckpt["g"], strict=False)
        del adapted_ckpt

    print("load saliency mask:", args.mask)
    if args.mask is not None:
        with torch.serialization.safe_globals([argparse.Namespace]):
            mask = torch.load(args.mask, map_location=lambda storage, loc: storage, weights_only=True)
    else: mask = None
    
    # load dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
    ])
    dataset = MultiResolutionDataset(args.path, transform, args.size)
    print(f"Dataset size: {len(dataset)}")
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=False,
    )
    
    # repulsion loss computer
    rloss = RepulsionLoss(
        gs_adapted, loss_type=args.loss_type, 
        alpha=args.alpha, beta=args.beta, exp_lambda=args.exp_lambda, 
        weight=args.rl_weight, dweight=args.dweight, do_scale=False
    )

    #-----------------------------------------------------------------
    os.makedirs(f"{args.outdir}/sample", exist_ok=True)
    os.makedirs(f"{args.outdir}/model", exist_ok=True)
    unlearn(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, rloss=rloss, mask=mask)
    # fid_score(args)
