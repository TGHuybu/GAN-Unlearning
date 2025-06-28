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


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def masking(args, generator, discriminator, num, device):
    generator.eval()
    discriminator.eval()
    requires_grad(generator, True)
    requires_grad(discriminator, False)

    top_ratio = args.top_ratio

    save_path = os.path.join(args.outdir, "mask.pt")

    # Initialize gradient accumulator
    gradients = {
        name: torch.zeros_like(param, device=device)
        for name, param in generator.named_parameters()
        if param.requires_grad
    }

    for _ in tqdm(range(num), desc="Computing gradients for mask"):
        # Fake image generation
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        fake_img, _ = generator(noise)

        fake_pred = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)
        
        generator.zero_grad()
        g_loss.backward()

        # Accumulate gradients
        with torch.no_grad():
            for name, param in generator.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.detach()

    # Take absolute value of gradients (saliency)
    for name in gradients:
        gradients[name] = torch.abs(gradients[name])

    # Build binary saliency mask based on top `top_ratio` elements
    all_elements = -torch.cat([g.flatten() for g in gradients.values()])
    threshold_index = int(len(all_elements) * top_ratio)

    positions = torch.argsort(all_elements) # index: rank, element: ORIGINAL INDEX
    ranks = torch.argsort(positions)        # index: ORIGINAL INDEX, element: rank

    start_index = 0
    mask_dict = {}
    for name, tensor in gradients.items():
        numel = tensor.numel()
        tensor_ranks = ranks[start_index:start_index + numel].reshape(tensor.shape)
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        mask_dict[name] = threshold_tensor
        start_index += numel

    # Save mask
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(mask_dict, save_path)
    print(f" Mask saved to: {save_path}")


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")
    
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument(
        "--num", type=int, default=100, help="gradient accumulation loops"
    )
    parser.add_argument(
        "--top_ratio", type=float, default=0.5, help="top saliency ranking ratio"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
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
        "--outdir",
        type=str,
        required=True,
        help="probability update interval of the adaptive augmentation",
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    if args.arch == 'stylegan2':
        from model import Generator, Discriminator
    elif args.arch == 'swagan':
        from swagan import Generator, Discriminator

    args.latent = 512
    args.n_mlp = 8
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

    print("load model:", args.ckpt)
    with torch.serialization.safe_globals([argparse.Namespace]):
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage, weights_only=True)
    generator.load_state_dict(ckpt["g"],strict=False)
    discriminator.load_state_dict(ckpt["d"],strict=False)
    g_ema.load_state_dict(ckpt["g_ema"],strict=False)

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan 2")

    os.makedirs(f"{args.outdir}/sample", exist_ok=True)
    os.makedirs(f"{args.outdir}/model", exist_ok=True)
    #args, loader, generator, discriminator, g_optim, d_optim, g_ema, device
    #  generator, discriminator, loader, device, save_path="mask_adapted.pt", top_ratio=0.5, num_batches=100
    masking(args, generator, discriminator, args.num, device)
