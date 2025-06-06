import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import glob
from utils.fid import (
    ImagePathDataset,
    get_activations_from_dataloader,
    calculate_frechet_distance,
)
import lpips
from itertools import product
from tqdm import tqdm

class MetricsCalculator:
    def __init__(self, datapath , max_dataset_size=100000, direction="AtoB", device='cpu'):
        self.device = device
        self.datapath=os.path.join(datapath, "results")
        self.direction=direction
        self.filename_list = os.listdir(self.datapath)
        self.file_list = [os.path.join(self.datapath, file) for file in self.filename_list]
        self.max_dataset_size = max_dataset_size
        target_name = "B" if direction=="AtoB" else "A"
        file_real = f"real_{target_name}*"
        file_fake = f"fake_{target_name}*"
        real_paths = [os.path.join(file, file_real) for file in self.file_list]
        fake_paths = [os.path.join(file, file_fake) for file in self.file_list]
        self.real_list = sorted([glob.glob(real_file)[0] for real_file in real_paths])[:self.max_dataset_size]
        self.fake_list = sorted([glob.glob(fake_file)[0] for fake_file in fake_paths])[:self.max_dataset_size]
        
        self.transform = transforms.ToTensor()
        
        self.real_dataset = ImagePathDataset(self.real_list, self.transform)
        self.fake_dataset = ImagePathDataset(self.fake_list, self.transform)
        
        self.real_dataloader = DataLoader(self.real_dataset, batch_size=1, shuffle=False, num_workers=0)
        self.fake_dataloader = DataLoader(self.fake_dataset, batch_size=1, shuffle=False, num_workers=0)

    def calculate_fid(self, verbose=False) -> float:
        real_activations = get_activations_from_dataloader(self.real_dataloader, verbose=verbose)
        fake_activations = get_activations_from_dataloader(self.fake_dataloader, verbose=verbose)
        mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
        mu_fake, sigma_fake = np.mean(fake_activations, axis=0), np.cov(fake_activations, rowvar=False)
        fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        # print(f"FID: {fid}")
        return fid


    def normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())

    def calculate_ssim(self, data_range=4095) -> float:
        """
        ensure that the real and fake images are paired
        """
        ssim_scores = []
        for real, fake in zip(self.real_dataloader, self.fake_dataloader):
            real = real.cpu().numpy().squeeze(0)
            fake = fake.cpu().numpy().squeeze(0)
            # real = self.normalize(real)
            # fake = self.normalize(fake)
            real = real * 2500
            fake = fake * 2500
            ssim_value = ssim(real, fake, data_range=data_range,channel_axis=0)
            ssim_scores.append(ssim_value)
        # print(f"SSIM: {np.mean(ssim_scores)}")
        return np.mean(ssim_scores), np.std(ssim_scores)

    def calculate_lpips(self) -> float:
        loss_fn = lpips.LPIPS(net='alex').to(self.device)
        lpips_scores = []
        for real, fake in tqdm(product(self.real_dataloader, self.fake_dataloader)):
            real = real.to(self.device)
            fake = fake.to(self.device)
            loss = loss_fn(real, fake)
            lpips_scores.append(loss.item())
        # print(f"LPIPS: {np.mean(lpips_scores)}")
        return np.mean(lpips_scores), np.std(lpips_scores)

    def calculate_psnr(self, data_range=4095) -> float:
        """
         ensure that the real and fake images are paired
        """
        psnr_scores = []
        for real, fake in zip(self.real_dataloader, self.fake_dataloader):
            # real = self.normalize(real)
            # fake = self.normalize(fake)
            real = real * 2500
            fake = fake * 2500
            mse = F.mse_loss(real, fake, reduction='none')
            psnr_value = 10 * torch.log10(data_range**2 / mse.mean())
            psnr_scores.append(psnr_value)
        # print(f"PSNR: {np.mean(psnr_scores)}")
        return np.mean(psnr_scores), np.std(psnr_scores)

    def calculate_all_metrics(self, verbose=False) -> dict:
        metrics = {
            'fid': self.calculate_fid(verbose=verbose),
            'ssim': self.calculate_ssim(),
            # 'lpips': self.calculate_lpips(),
            'psnr': self.calculate_psnr()
        }
        return metrics 