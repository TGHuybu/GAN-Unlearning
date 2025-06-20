import argparse
import os

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from torchvision import utils
from model import Generator
from tqdm import tqdm


class FacialAttributeClassifier:
    def __init__(self, model_path, attributes_file):
        # Khởi tạo model
        self.model = models.resnext50_32x4d(pretrained=False)
        self.model.fc = nn.Linear(2048, 40)
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.eval()
        
        # Chuyển model sang GPU nếu có
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Chuẩn bị transform ảnh
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.ToTensor()
        ])
        
        # # Load danh sách thuộc tính
        # with open(attributes_file, 'r') as f:
        #     self.attributes = f.readlines()[2].split()  # Lấy header từ file
        self.attributes = np.loadtxt(attributes_file, dtype=str)


    def flag_attr(self, image_batch, target_class):
        img_tensor = self.transform(image_batch).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
        
        target_idx = np.where(self.attributes == target_class)[0]
        output = output.cpu().numpy()
        flags = (output[:, target_idx] >= 0).flatten()
        
        return flags


def find_target_latent(args, g_ema, classifier, device, mean_latent):
    n_found = 0
    latent_list = []
    while n_found < args.sample:
        print(f">> extracted {n_found}")
        with torch.no_grad():
            sample_z = torch.randn(args.sample, args.latent, device=device)
            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent
            )

            flags = classifier.flag_attr(sample, args.neg_class)
            sample_z = sample_z.cpu().numpy()
            sample_z_target = sample_z[flags]
            if sample_z_target.shape[0] != 0:
                print(sample_z_target.shape)
                latent_list.append(torch.from_numpy(sample_z_target))
                n_found += np.sum(flags.astype(int))

    target_latents = torch.cat(latent_list, dim=0)
    return target_latents[0:args.sample]


def generate(args, g_ema, input_latent, device, mean_latent, name="sample"):
    with torch.no_grad():
        g_ema.eval()
        sample_z = input_latent.to(device)
        sample, _ = g_ema(
            [sample_z], truncation=args.truncation, truncation_latent=mean_latent
        )

        utils.save_image(
            sample,
            f"{args.path}/{name}.png",
            nrow=int(args.sample ** 0.5),
            normalize=True
        )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--ckpt_original",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--ckpt_unlearn_l2inv",
        type=str,
        default="None",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--ckpt_unlearn_l2neg",
        type=str,
        default="None",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--ckpt_unlearn_l2exp",
        type=str,
        default="None",
        help="path to the model checkpoint",
    )
    parser.add_argument("--classifier", type=str, required=True, help="path to the classifier")
    parser.add_argument("--attr_list", type=str, required=True, help="path to the attribute list")
    parser.add_argument(
        "--neg_class", 
        type=str, default="Eyeglasses", help="undesired attribute class (see file attr_names.txt)"
    )

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=16,
        help="number of samples to be generated for each image",
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to the image directory to be generated",
    )

    args = parser.parse_args()

    # Original generator
    args.latent = 512
    args.n_mlp = 8
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.ckpt_original, weights_only=True)
    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    # Feature classifier
    classifier = FacialAttributeClassifier(
        model_path=args.classifier,
        attributes_file=args.attr_list
    )

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    latents = find_target_latent(args, g_ema, classifier, device, mean_latent)

    os.makedirs(args.path, exist_ok=True)

    generate(args, g_ema, latents, device, mean_latent, name="original")
    del g_ema, checkpoint

    if args.ckpt_unlearn_l2inv != "None":
        try:
            g_l2inv = Generator(
                args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
            ).to(device)
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint_l2inv = torch.load(args.ckpt_unlearn_l2inv, weights_only=True)
            g_l2inv.load_state_dict(checkpoint_l2inv["g_ema"], strict=False)
            generate(args, g_l2inv, latents, device, mean_latent, name="unlearned_l2inv")
            del g_l2inv, checkpoint_l2inv
        except Exception as e:
            print(e)
        
    if args.ckpt_unlearn_l2neg != "None":
        try:
            g_l2neg = Generator(
                args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
            ).to(device)
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint_l2neg = torch.load(args.ckpt_unlearn_l2neg, weights_only=True)
            g_l2neg.load_state_dict(checkpoint_l2neg["g_ema"], strict=False)
            generate(args, g_l2neg, latents, device, mean_latent, name="unlearned_l2neg")
            del g_l2neg, checkpoint_l2neg
        except Exception as e:
            print(e)
        
    if args.ckpt_unlearn_l2exp != "None":
        try:
            g_l2exp = Generator(
                args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
            ).to(device)
            with torch.serialization.safe_globals([argparse.Namespace]):
                checkpoint_l2exp = torch.load(args.ckpt_unlearn_l2exp, weights_only=True)
            g_l2exp.load_state_dict(checkpoint_l2exp["g_ema"], strict=False)
            generate(args, g_l2exp, latents, device, mean_latent, name="unlearned_l2exp")
            del g_l2exp, checkpoint_l2exp
        except Exception as e:
            print(e)