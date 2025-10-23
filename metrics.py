import math
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

def cv2PSNR(image1: np.ndarray, image2: np.ndarray) -> float:
    '''
    image1: H, W, C nd.array
    image2: H, W, C nd.array
    '''
    return cv2.PSNR(image1, image2)

def skimageSSIM(image1: np.ndarray, image2: np.ndarray) -> float:
    '''
    image1: H, W, C nd.array
    image2: H, W, C nd.array
    '''
    return ssim(image1, image2, channel_axis=2)

# TEST FUNCTIONS
if __name__ == "__main__":
    from datareader import ImageReader
    from torch.utils.data import Dataset, DataLoader
    from einops import rearrange
    from utils import unnormalize
    PSNRs = []
    SSIMs = []
    dataset = ImageReader(Mode='test') 
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True) # test X 3, H, W Y H, W, 3
    for idx, (noisy, original) in enumerate(dataloader):
        noisy = rearrange(noisy, 'b c h w -> b h w c').numpy()
        for i in range(original.shape[0]):
            PSNRs.append(cv2PSNR(original[i].numpy(), (unnormalize(noisy[i])*255).astype(np.uint8)))
            SSIMs.append(skimageSSIM(original[i].numpy(), (unnormalize(noisy[i])*255).astype(np.uint8)))
    print(f"Average PSNR: {sum(PSNRs) / len(PSNRs)}")
    print(f"Average SSIM: {sum(SSIMs) / len(SSIMs)}")
    print(PSNRs)
    print(SSIMs)
