import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from scipy import linalg
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from typing import List, Tuple, Union
import lpips
from torch.utils.data import DataLoader
from tqdm import tqdm

class MetricsCalculator:
    def __init__(self, device='cuda:0'):
        self.device = device
        # 初始化Inception模型用于FID计算
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.fc = nn.Identity()  # 移除最后的全连接层
        self.inception_model = self.inception_model.to(device)
        self.inception_model.eval()
        
        # 初始化LPIPS模型
        self.lpips_model = lpips.LPIPS(net='alex').to(device)
        
        # 定义图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def calculate_fid(self, real_dataloader: DataLoader, fake_dataloader: DataLoader) -> float:
        """
        calculate FID score (unpaired version)
        Args:
            real_dataloader: real images dataloader
            fake_dataloader: fake images dataloader
        Returns:
            FID score
        """
        real_features = []
        fake_features = []
        
        # 提取真实图像特征
        with torch.no_grad():
            for real_images in tqdm(real_dataloader, desc="extract real features"):
                if isinstance(real_images, (list, tuple)):
                    real_images = real_images[0]
                real_images = real_images.to(self.device)
                real_images = self.preprocess(real_images)
                features = self.inception_model(real_images)
                real_features.append(features.cpu())
            
            # 提取生成图像特征
            for fake_images in tqdm(fake_dataloader, desc="extract fake features"):
                if isinstance(fake_images, (list, tuple)):
                    fake_images = fake_images[0]
                fake_images = fake_images.to(self.device)
                fake_images = self.preprocess(fake_images)
                features = self.inception_model(fake_images)
                fake_features.append(features.cpu())
        
        # 合并所有特征
        real_features = torch.cat(real_features, dim=0)
        fake_features = torch.cat(fake_features, dim=0)
        
        # 计算均值和协方差
        mu_real = real_features.mean(0)
        mu_fake = fake_features.mean(0)
        sigma_real = torch.cov(real_features.T)
        sigma_fake = torch.cov(fake_features.T)
        
        # 计算FID
        ssdiff = torch.sum((mu_real - mu_fake) ** 2.0)
        covmean = torch.matrix_power(sigma_real @ sigma_fake, 0.5)
        
        fid = ssdiff + torch.trace(sigma_real + sigma_fake - 2.0 * covmean)
        return fid.item()

    def calculate_ssim(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        计算SSIM分数
        Args:
            real_images: 真实图像张量 [B, C, H, W]
            fake_images: 生成图像张量 [B, C, H, W]
        Returns:
            SSIM分数
        """
        # 转换为numpy数组
        real_np = real_images.cpu().numpy()
        fake_np = fake_images.cpu().numpy()
        
        ssim_scores = []
        for real, fake in zip(real_np, fake_np):
            # 转换为[0, 1]范围
            real = (real + 1) / 2
            fake = (fake + 1) / 2
            # 计算SSIM
            score = ssim(real, fake, data_range=1.0, channel_axis=0)
            ssim_scores.append(score)
        
        return np.mean(ssim_scores)

    def calculate_lpips(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        计算LPIPS分数
        Args:
            real_images: 真实图像张量 [B, C, H, W]
            fake_images: 生成图像张量 [B, C, H, W]
        Returns:
            LPIPS分数
        """
        with torch.no_grad():
            lpips_score = self.lpips_model(real_images, fake_images)
        return lpips_score.mean().item()

    def calculate_psnr(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """
        计算PSNR分数
        Args:
            real_images: 真实图像张量 [B, C, H, W]
            fake_images: 生成图像张量 [B, C, H, W]
        Returns:
            PSNR分数
        """
        # 转换为[0, 1]范围
        real = (real_images + 1) / 2
        fake = (fake_images + 1) / 2
        
        mse = F.mse_loss(real, fake)
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()

    def calculate_all_metrics(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> dict:
        """
        计算所有指标
        Args:
            real_images: 真实图像张量 [B, C, H, W]
            fake_images: 生成图像张量 [B, C, H, W]
        Returns:
            包含所有指标的字典
        """
        metrics = {
            'fid': self.calculate_fid(real_images, fake_images),
            'ssim': self.calculate_ssim(real_images, fake_images),
            'lpips': self.calculate_lpips(real_images, fake_images),
            'psnr': self.calculate_psnr(real_images, fake_images)
        }
        return metrics 