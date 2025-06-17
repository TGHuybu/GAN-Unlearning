import torch
import numpy as np
from pytorch_fid import fid_score
from generate import generate
from model import Generator
import argparse
from lpips import LPIPS
from PIL import Image
from torchvision import transforms

device = torch.device('cuda')

def fid_cal(args, g_ema, device, mean_latent):
    fid_values = []
    path2 = '/media02/lhthai/stylegan2-pytorch/FaceTest/cartoontest/'    
    path1 = '/media02/lhthai/stylegan2-pytorch/generated/'

    for i in range(args.num_samples):
        # Generate images
        generate(args, g_ema, device, mean_latent)
        fid_value = fid_score.calculate_fid_given_paths([path1, path2], batch_size=1, device=device, dims=2048)
        fid_values.append(fid_value)

    mean_fid = np.mean(fid_values)
    std_fid = np.std(fid_values)
    std_error = std_fid / np.sqrt(len(fid_values))
    print(f"Mean FID: {mean_fid:.4f} ± {std_error:.4f}")



def lpips_cal(args, g_ema, device, mean_latent):
    # Khởi tạo LPIPS loss function và chuyển sang device
    loss_fn = LPIPS(net='vgg').to(device)
    
    # Định nghĩa transform: chuyển ảnh sang tensor và chuẩn hóa về khoảng [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    lpips_values = []
    args.pics = 1
    for i in range(args.num_samples):
        # Sinh ảnh đầu tiên
        args.path = '/media02/lhthai/stylegan2-pytorch/lpips/img1'
        generate(args, g_ema, device, mean_latent)
        

        # Sinh ảnh thứ hai
        args.path = '/mnt/d/stylegan2-pytorch/lpips/img2'
        generate(args, g_ema, device, mean_latent)
        

        # Tính toán khoảng cách LPIPS
        d = loss_fn(img1, img2)
        lpips_values.append(d.item())

    # Tính giá trị trung bình và sai số chuẩn
    mean_lpips = np.mean(lpips_values)
    std_lpips = np.std(lpips_values)
    std_error = std_lpips / np.sqrt(len(lpips_values))
    print(f"Mean LPIPS: {mean_lpips:.4f} ± {std_error:.4f}")


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=1024, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=20, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stylegan2-ffhq-config-f.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path to the image directory to be generated",
    )
    parser.add_argument(
        "--path_lpips1",
        type=str,
        default="/lpips/img1",
        help="path to the image directory to be generated",
    )
    parser.add_argument(
        "--path_lpips2",
        type=str,
        default="/lpips/img2",
        help="path to the image directory to be generated",
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt, weights_only=False)

    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    fid_cal(args, g_ema, device, mean_latent)
    lpips_cal(args, g_ema, device, mean_latent)


