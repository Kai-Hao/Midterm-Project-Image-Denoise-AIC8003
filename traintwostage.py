from datareader import ImageReaderWithNoiseMap
from torch.utils.data import Dataset, DataLoader
from model import DnCNN, Restormer
from torch import nn, optim
from einops import rearrange
from metrics import cv2PSNR, skimageSSIM
from utils import unnormalize
from tqdm import tqdm
import numpy as np
import torch
import os

# HYPERPARAMETERS
EPOCHS            = 70
BATCH_SIZE        = 4
LEARNING_RATE     = 1e-3
END_LEARNING_RATE = 1e-5
WEIGHT_DECAY      = 1e-4
MODEL_NAME        = 'Restormer'
SAVE_PATH         = './checkpoints/'
os.makedirs(SAVE_PATH, exist_ok=True)

# DATASET AND DATALOADER
train_dataset = ImageReaderWithNoiseMap(Mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataset = ImageReaderWithNoiseMap(Mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = Restormer() # DnCNN/Restormer
# pretrained_dict = torch.load('./pretrained/gaussian_color_denoising_blind.pth')
# model.load_state_dict(pretrained_dict['params'])
model = model.cuda()

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=70, T_mult=2, eta_min=END_LEARNING_RATE)
best_psnr = 0.0
best_ssim = 0.0

for epoch in range(EPOCHS):
    min_loss = float('inf')
    model.train()
    train_loss = 0.0
    for noisy, original in tqdm(train_dataloader):
        noisy = noisy.cuda()
        original = original.cuda()
        optimizer.zero_grad()
        output = model(noisy)
        loss = criterion(output, original)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * noisy.size(0)
    train_loss /= len(train_dataloader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}")
    scheduler.step()
    # Validation
    PSNRs, SSIMs = [], []
    model.eval()
    with torch.no_grad():
        for noisy, original in tqdm(test_dataloader):
            noisy = noisy.cuda()
            original = original.cuda()
            output = model(noisy)
            for i in range(output.size(0)):
                rearranged_output = rearrange(output[i], 'c h w -> h w c').cpu().numpy()
                rearranged_output = unnormalize(rearranged_output)
                rearranged_output = np.clip(rearranged_output, 0, 1)
                rearranged_output = (rearranged_output * 255).astype(np.uint8)
                psnr = cv2PSNR(rearranged_output, original[i].cpu().numpy())
                ssim = skimageSSIM(rearranged_output, original[i].cpu().numpy())
                PSNRs.append(psnr)
                SSIMs.append(ssim)
    avg_psnr = np.mean(PSNRs)
    avg_ssim = np.mean(SSIMs)
    print(f"Epoch [{epoch+1}/{EPOCHS}], Test PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    if avg_psnr > best_psnr:
        best_psnr = avg_psnr
        best_ssim = avg_ssim
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, f'best_{MODEL_NAME}_model_with_pixelmodel.pth'))
        print(f"Saved Best Model with PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}")
print('final best psnr:', best_psnr, 'ssim:', best_ssim)
