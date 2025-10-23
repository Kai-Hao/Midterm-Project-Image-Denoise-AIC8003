from datareader import ImageReader
from torch.utils.data import Dataset, DataLoader
from model import DnCNN, Restormer, MedianFilter, GaussianFilter, BilateralFilter
from torch import nn, optim
from einops import rearrange
from metrics import cv2PSNR, skimageSSIM
from utils import unnormalize
from tqdm import tqdm
import numpy as np
import torch
import os
import cv2

# HYPERPARAMETERS
EPOCHS             = 70
BATCH_SIZE         = 4
LEARNING_RATE      = 1e-3
END_LEARNING_RATE  = 1e-5
WEIGHT_DECAY       = 1e-4
MODEL_NAME         = 'TwoStage_median_Restormer'
TRADITIONAL_METHOD = 'median'  # 'median', 'gaussian', 'bilateral'
SAVE_PATH          = './checkpoints/'
os.makedirs(SAVE_PATH, exist_ok=True)

# DATASET AND DATALOADER
train_dataset = ImageReader(Mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = ImageReader(Mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

if TRADITIONAL_METHOD == 'median':
    traditional_model = MedianFilter(kernel_size=3).cuda()
elif TRADITIONAL_METHOD == 'gaussian':
    traditional_model = GaussianFilter(kernel_size=3, sigma=1.0).cuda()
elif TRADITIONAL_METHOD == 'bilateral':
    traditional_model = BilateralFilter(kernel_size=3, sigma_spatial=1.0, sigma_intensity=0.1).cuda()
else:
    traditional_model = None

deep_model = Restormer()  # DnCNN/Restormer
# pretrained_dict = torch.load('./pretrained/gaussian_color_denoising_blind.pth')
# deep_model.load_state_dict(pretrained_dict['params'])
deep_model = deep_model.cuda()

def apply_traditional_preprocessing(noisy_batch):
    batch_size = noisy_batch.size(0)
    preprocessed_batch = torch.zeros_like(noisy_batch)
    
    for i in range(batch_size):
        noisy_np = rearrange(noisy_batch[i], 'c h w -> h w c').cpu().numpy()
        noisy_np = unnormalize(noisy_np)
        
        with torch.no_grad():
            noisy_tensor = noisy_batch[i:i+1].cuda()
            preprocessed_tensor = traditional_model(noisy_tensor)
            preprocessed_np = rearrange(preprocessed_tensor[0], 'c h w -> h w c').cpu().numpy()
            preprocessed_np = unnormalize(preprocessed_np)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        preprocessed_np = (preprocessed_np - mean) / std
        preprocessed_batch[i] = torch.tensor(rearrange(preprocessed_np, 'h w c -> c h w'), dtype=torch.float32)
    
    return preprocessed_batch

criterion = nn.L1Loss()
optimizer = optim.Adam(deep_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=70, T_mult=2, eta_min=END_LEARNING_RATE)

best_psnr = 0.0
best_ssim = 0.0

for epoch in range(EPOCHS):
    # Training Phase
    deep_model.train()
    train_loss = 0.0
    
    for noisy, original in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Training"):
        noisy = noisy.cuda()
        original = original.cuda()
        
        preprocessed = apply_traditional_preprocessing(noisy).cuda()

        optimizer.zero_grad()
        output = deep_model(preprocessed)
        loss = criterion(output, original)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * noisy.size(0)
    
    train_loss /= len(train_dataloader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}")
    
    scheduler.step()
    
    PSNRs, SSIMs = [], []
    deep_model.eval()
    
    with torch.no_grad():
        for noisy, original in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS} - Validation"):
            noisy = noisy.cuda()
            original = original.cuda()

            preprocessed = apply_traditional_preprocessing(noisy).cuda()
            output = deep_model(preprocessed)

            for i in range(output.size(0)):
                rearranged_output = rearrange(output[i], 'c h w -> h w c').cpu().numpy()
                rearranged_output = unnormalize(rearranged_output)
                rearranged_output = np.clip(rearranged_output, 0, 1)
                rearranged_output = (rearranged_output * 255).astype(np.uint8)
                
                original_np = original[i].cpu().numpy().astype(np.uint8)
                
                psnr = cv2PSNR(original_np, rearranged_output)
                ssim = skimageSSIM(original_np, rearranged_output)
                PSNRs.append(psnr)
                SSIMs.append(ssim)
    
    avg_psnr = np.mean(PSNRs)
    avg_ssim = np.mean(SSIMs)
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Test PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        best_ssim = avg_ssim
        torch.save(deep_model.state_dict(), 
                  os.path.join(SAVE_PATH, f'best_{MODEL_NAME}_twostage.pth'))
        print(f"Saved Best Model with PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}")

print(f'Best Result - PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}')
print(f'save path: {SAVE_PATH}best_{MODEL_NAME}_twostage.pth')