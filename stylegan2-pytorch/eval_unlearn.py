import argparse
import os
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import inception 
from dataset import MultiResolutionDataset
from scipy.linalg import sqrtm


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


    def compute_n_target(self, image_batch, target_class):
        img_tensor = self.transform(image_batch).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
        
        target_idx = np.where(self.attributes == target_class)[0]
        output = output.cpu().numpy()
        predictions = (output[:, target_idx] >= 0).astype(int).flatten()
        
        return np.sum(predictions)
    

@torch.no_grad()
def extract_features(loader, extractor, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = extractor(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to("cpu"))

    features = torch.cat(feature_list, 0)

    return features


def calculate_fid(features1, features2):
	# calculate activations
	# calculate mean and covariance statistics
	mu1, sigma1 = features1.mean(axis=0), np.cov(features1, rowvar=False)
	mu2, sigma2 = features2.mean(axis=0), np.cov(features2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

    # mu1 = np.atleast_1d(mu1)
    # mu2 = np.atleast_1d(mu2)

    # sigma1 = np.atleast_2d(sigma1)
    # sigma2 = np.atleast_2d(sigma2)

    # diff = mu1 - mu2

    # # Product might be almost singular
    # covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # if not np.isfinite(covmean).all():
    #     msg = ('fid calculation produces singular product; '
    #            'adding %s to diagonal of cov estimates') % eps
    #     print(msg)
    #     offset = np.eye(sigma1.shape[0]) * eps
    #     covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # # Numerical error might give slight imaginary component
    # if np.iscomplexobj(covmean):
    #     if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(covmean.imag))
    #         raise ValueError('Imaginary component {}'.format(m))
    #     covmean = covmean.real

    # tr_covmean = np.trace(covmean)

    # return (diff.dot(diff) + np.trace(sigma1)
    #         + np.trace(sigma2) - 2 * tr_covmean)


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser(description="StyleGAN2 adapter")

    #--- REQUIRED ARGS
    parser.add_argument("--original_gen", type=str, required=True, help="path to the ORIGINAL model")
    parser.add_argument("--unlearned_gen", type=str, required=True, help="path to the UNLEARNED model")
    parser.add_argument("--retrained_gen", type=str, required=True, help="path to the RETRAINED model (using only desired data)")
    parser.add_argument("--classifier", type=str, required=True, help="path to the classifier")
    parser.add_argument("--path", type=str, required=True, help="path to the lmdb dataset (removed undesired class)")
    parser.add_argument("--attr_list", type=str, required=True, help="path to the attribute list")

    #--- OPTIONAL ARGS
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    # parser.add_argument(
    #     "--iter", type=int, default=200, help="total sampling iterations"
    # )
    # parser.add_argument(
    #     "--n_sample", type=int, default=16, help="number of samples per iteration"
    # )
    parser.add_argument(
        "--batch", default=16, type=int, help="batch size for inception networks"
    )
    parser.add_argument(
        "--neg_class", 
        type=str, default="Eyeglasses", help="undesired attribute class (see file attr_names.txt)"
    )
    # parser.add_argument(
    #     "--dims",
    #     type=int,
    #     default=2048,
    #     choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
    #     help=(
    #         "Dimensionality of Inception features to use. "
    #         "By default, uses pool3 features"
    #     ),
    # )
    parser.add_argument(
        "--flip", action="store_true", help="apply random flipping to real images"
    )

    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
        transforms.ToTensor()
    ])
    transform_inception = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    
    ## Original generated
    dset_og = MultiResolutionDataset(args.original_gen, transform=transform, resolution=args.size)
    loader_og = DataLoader(dset_og, batch_size=args.batch, num_workers=2)

    ## Unlearned generated
    dset_ul = MultiResolutionDataset(args.unlearned_gen, transform=transform, resolution=args.size)
    loader_ul = DataLoader(dset_ul, batch_size=args.batch, num_workers=2)

    ## Retrained generated
    dset_rt = MultiResolutionDataset(args.retrained_gen, transform=transform_inception, resolution=args.size)
    loader_rt = DataLoader(dset_rt, batch_size=args.batch, num_workers=2)

    ## Classifier model
    classifier = FacialAttributeClassifier(
        model_path=args.classifier,
        attributes_file=args.attr_list
    )

    ## Filtered original dataset (removed undesired features/classes)
    dset_filtered = MultiResolutionDataset(args.path, transform=transform_inception, resolution=args.size)
    loader_filtered = DataLoader(dset_filtered, batch_size=args.batch, num_workers=2)

    #--- COMPUTE PUL
    ## Number of undesired samples (original)
    print("## Calculating number of undesired samples (original)")
    n_undesired_og = 0
    for imgs_og in loader_og:
        imgs_og = imgs_og.to(device)
        n_undesired_og += classifier.compute_n_target(imgs_og, target_class=args.neg_class)

    ## Number of undesired samples (unlearned)
    print("## Calculating number of undesired samples (unlearned)")
    n_undesired_ul = 0
    for imgs_ul in loader_ul:
        imgs_ul = imgs_ul.to(device)
        n_undesired_ul += classifier.compute_n_target(imgs_ul, target_class=args.neg_class)

    # clean up memory
    del dset_og, loader_og, dset_ul, loader_ul
    
    #--- Calculate PUL
    pul = ((n_undesired_og - n_undesired_ul)/n_undesired_og)*100
    print(f"-- OG: {n_undesired_og}; UL: {n_undesired_ul}")
    print(f">> Percentage of UnLearning (PUL): {pul}")

    #--- COMPUTE FID + Ret-FID
    # need to normalize before feeding to inception
    dset_ul_norm = MultiResolutionDataset(args.unlearned_gen, transform=transform_inception, resolution=args.size)
    loader_ul_norm = DataLoader(dset_ul_norm, batch_size=args.batch, num_workers=2)

    inception_model = inception.InceptionV3([3], normalize_input=False)
    inception_model = nn.DataParallel(inception_model).eval().to(device)

    ## FID (unlearned vs data)
    features_dset_filtered = extract_features(loader_filtered, inception_model, device).numpy()
    features_ul = extract_features(loader_ul_norm, inception_model, device).numpy()
    fid = calculate_fid(features_dset_filtered, features_ul)
    print(f">> Frechet Inception Distance (FID): {fid}")

    ## Ret-FID (unlearned vs retrained)
    features_rt = extract_features(loader_rt, inception_model, device).numpy()
    ret_fid = calculate_fid(features_rt, features_ul)
    print(f">> Retraining FID (Ret-FID): {ret_fid}")
