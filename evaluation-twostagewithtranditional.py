import cv2
import torch
from model import DnCNN, Restormer, MedianFilter, GaussianFilter, BilateralFilter
from datareader import ImageReader
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from metrics import cv2PSNR as PSNR, skimageSSIM as SSIM
from utils import unnormalize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os

TRADITIONAL_METHOD = "bilateral"  # "median", "gaussian", "bilateral"
DEEP_MODEL = "Restormer"  # "DnCNN", "Restormer"
model_name = f'TwoStage_bilateral_Restormer_finetune'
model_path = f"./checkpoints/best_TwoStage_{TRADITIONAL_METHOD}_{DEEP_MODEL}_finetune_twostage.pth"

if os.path.exists(f"./Visualize/{model_name}"):
    shutil.rmtree(f"./Visualize/{model_name}")
os.makedirs(f"./Visualize/{model_name}", exist_ok=True)

ImageDatasets = ImageReader(Mode='test')
ImageDataloader = DataLoader(ImageDatasets, batch_size=1, shuffle=False)

if TRADITIONAL_METHOD == 'median':
    TraditionalModel = MedianFilter(kernel_size=3).cuda()
elif TRADITIONAL_METHOD == 'gaussian':
    TraditionalModel = GaussianFilter(kernel_size=3, sigma=1.0).cuda()
elif TRADITIONAL_METHOD == 'bilateral':
    TraditionalModel = BilateralFilter(kernel_size=3, sigma_spatial=1.0, sigma_intensity=0.1).cuda()

if DEEP_MODEL == "DnCNN":
    DeepModel = DnCNN().cuda()
elif DEEP_MODEL == "Restormer":
    DeepModel = Restormer().cuda()

checkpoint = torch.load(model_path)
DeepModel.load_state_dict(checkpoint)
print(f"Successfully loaded model: {model_path}")


TraditionalModel.eval()
DeepModel.eval()

def apply_traditional_preprocessing(noisy_batch):
    batch_size = noisy_batch.size(0)
    preprocessed_batch = torch.zeros_like(noisy_batch)
    
    for i in range(batch_size):
        noisy_np = rearrange(noisy_batch[i], 'c h w -> h w c').cpu().numpy()
        noisy_np = unnormalize(noisy_np)
        
        with torch.no_grad():
            noisy_tensor = noisy_batch[i:i+1].cuda()
            preprocessed_tensor = TraditionalModel(noisy_tensor)
            preprocessed_np = rearrange(preprocessed_tensor[0], 'c h w -> h w c').cpu().numpy()
            preprocessed_np = unnormalize(preprocessed_np)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        preprocessed_np = (preprocessed_np - mean) / std
        preprocessed_batch[i] = torch.tensor(rearrange(preprocessed_np, 'h w c -> c h w'), dtype=torch.float32)
    
    return preprocessed_batch

PSNRs = []
SSIMs = []

for idx, (noisy, original) in enumerate(tqdm(ImageDataloader)):
    noisy = noisy.cuda()
    
    with torch.no_grad():
        preprocessed = apply_traditional_preprocessing(noisy).cuda()
        denoised = DeepModel(preprocessed)
    denoised = denoised.cpu().squeeze(0)
    original = original.squeeze(0).numpy().astype(np.uint8)
    denoised = (unnormalize(rearrange(denoised, 'c h w -> h w c').numpy()) * 255).astype(np.uint8)

    PSNR_value = round(PSNR(original, denoised), 3)
    SSIM_value = round(SSIM(original, denoised), 3)
    PSNRs.append(PSNR_value)
    SSIMs.append(SSIM_value)
    
    MAEMAP = np.abs(original.astype(np.float32) - denoised.astype(np.float32))
    MAEMAP = MAEMAP.astype(np.uint8)
    MAEMAP_colored = cv2.applyColorMap(MAEMAP, cv2.COLORMAP_INFERNO)
    MAEMAP = cv2.cvtColor(MAEMAP_colored, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite(f"./Visualize/{model_name}/MAE_MAP_{idx+1}_{PSNR_value}_{SSIM_value}.png", 
                cv2.cvtColor(MAEMAP, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"./Visualize/{model_name}/Denoised_{idx+1}_{PSNR_value}_{SSIM_value}.png", 
                cv2.cvtColor(denoised, cv2.COLOR_RGB2BGR))
    
    combined = np.hstack((denoised, original, MAEMAP))
    
    # 4 x 5 subplot
    plt.subplot(5, 4, idx + 1)
    plt.imshow(combined)
    plt.axis('off')
    plt.title(f"PSNR: {PSNR_value:.2f}\nSSIM: {SSIM_value:.4f}", fontsize=6)

avg_psnr = sum(PSNRs) / len(PSNRs)
avg_ssim = sum(SSIMs) / len(SSIMs)

plt.suptitle(f"{model_name} - Average PSNR: {avg_psnr:.3f} | Average SSIM: {avg_ssim:.4f}", 
             fontsize=10, fontweight='bold', y=0.99)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig(f"./Visualize/{model_name}/{model_name}_results.png")
plt.show()

print(f"Average PSNR: {avg_psnr:.3f}")
print(f"Average SSIM: {avg_ssim:.4f}")