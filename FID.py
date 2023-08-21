import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3
import os
from PIL import Image
import numpy as np
from scipy.linalg import sqrtm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Load the InceptionV3 model
inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model = inception_model.eval()

# Functions to compute FID
def compute_statistics(images, model):
    images = images.to(device)
    with torch.no_grad():
        preds = model(images).detach().cpu().numpy()
    mu = np.mean(preds, axis=0)
    sigma = np.cov(preds, rowvar=False)
    return mu, sigma

def compute_fid(mu1, sigma1, mu2, sigma2):
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_fid(real_path, generated_path):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])
    
    real_dataset = CustomImageDataset(real_path, transform=transform)
    generated_dataset = CustomImageDataset(generated_path, transform=transform)
    
    real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    generated_loader = DataLoader(generated_dataset, batch_size=32, shuffle=False)

    # Compute statistics for real and generated images
    mu_real, sigma_real = compute_statistics(next(iter(real_loader)), inception_model)
    mu_generated, sigma_generated = compute_statistics(next(iter(generated_loader)), inception_model)

    # Compute FID
    fid_value = compute_fid(mu_real, sigma_real, mu_generated, sigma_generated)
    return fid_value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute FID between two sets of images.')
    parser.add_argument('--real_path', type=str, required=True, help='Path to the directory of real images')
    parser.add_argument('--generated_path', type=str, required=True, help='Path to the directory of generated images')
    parser.add_argument('--output', type=str, required=True, help='Output file to store FID value')

    args = parser.parse_args()
    fid_value = calculate_fid(args.real_path, args.generated_path)
    
    with open(args.output, 'w') as f:
        f.write(str(fid_value))
    
    print("FID:", fid_value)
