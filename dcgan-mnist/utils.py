import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm


def generate_folder(generator, nz, device, folderpath="generated", examples=25, no_save=True):
    noise = torch.randn(examples, nz, 1, 1, device=device)
    generator.eval()
    with torch.no_grad():
        generated_images = generator(noise).detach().cpu()
        torch.cuda.empty_cache()

    if no_save: return generated_images

    os.makedirs(folderpath, exist_ok=True)
    for i in range(generated_images.shape[0]):
        file_name = f"{i}.jpg"
        image = generated_images[i].permute(1, 2, 0)
        image = image.numpy()
        image = image.reshape(image.shape[0], image.shape[1])
        plt.imsave(os.path.join(folderpath, file_name), image, cmap="gray")
        
    
def get_features(extractor, img_batch, device):
    transformation = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    img_batch_transformed = transformation(img_batch)

    extractor.eval()
    with torch.no_grad():
        features = extractor.extract_features(img_batch_transformed.to(device)).detach().cpu()
        torch.cuda.empty_cache()

    return features.numpy()


def classify(classifier, img_batch, device):
    transformation = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    img_batch_transformed = transformation(img_batch)

    classifier.eval()
    with torch.no_grad():
        _, predictions = torch.max(classifier(img_batch_transformed.to(device)), 1)
        torch.cuda.empty_cache()

    return predictions.detach().cpu().numpy()


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


def compute_FIDs(model1, classifier, nz, device, model2=None, dset_loader=None, n_iters=10):
    fids = []
    for i in range(n_iters):
        if dset_loader is None:
            generated_1 = generate_folder(model1, nz, device, examples=20000)
            generated_2 = generate_folder(model2, nz, device, examples=20000)
            features_1 = get_features(classifier, generated_1, device)
            features_2 = get_features(classifier, generated_2, device)
            fid = calculate_fid(features_1, features_2)
            fids.append(fid)
        else:
            features_original_data_list = []
            count = 0
            for i, (images, labels) in enumerate(dset_loader):
                # if count >= 20000: break
                features_original_data_list.append(get_features(classifier, images, device))
                count += images.size(0)
            features_original_data = np.concatenate(features_original_data_list, axis=0)

            generated = generate_folder(model1, nz, device, examples=20000)
            features_generated = get_features(classifier, generated, device)
            fid = calculate_fid(features_original_data, features_generated)
            fids.append(fid)
        torch.cuda.empty_cache()

    return np.array(fids)


def compute_PULs(original_model, unlearned_model, classifier, negative_class, nz, device, n_iters=10):
    puls = []
    for i in range(n_iters):
        generated_original = generate_folder(original_model, nz, device, examples=20000)
        generated_unlearned = generate_folder(unlearned_model, nz, device, examples=20000)
        classifications_original = classify(classifier, generated_original, device)
        classifications_unlearned= classify(classifier, generated_unlearned, device)

        n_negative_original = np.sum(classifications_original == negative_class)
        n_negative_unlearned = np.sum(classifications_unlearned == negative_class)

        pul = ((n_negative_original - n_negative_unlearned)/n_negative_original) * 100
        puls.append(pul)

    return np.array(puls)