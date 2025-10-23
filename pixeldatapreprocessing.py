import os
import cv2
import numpy as np
from utils import get_noise_index

original_Folder = './Midterm100/original/'
noise_Folder = './Midterm100/noisy/'
noise_mask_Folder = './Midterm100/noiseMask/'
kernel_size = 3
outputTrainX = './Midterm100/pixeltrain/pixel/'
outputTrainY = './Midterm100/pixeltrain/label/'
outputTestX = './Midterm100/pixeltest/pixel/'
outputTestY = './Midterm100/pixeltest/label/'
os.makedirs(outputTrainX, exist_ok=True)
os.makedirs(outputTrainY, exist_ok=True)    
os.makedirs(outputTestX, exist_ok=True)
os.makedirs(outputTestY, exist_ok=True)
for i in range(1, 101, 1):
    if i == 82:
        continue
    
    noise_mask_path = f'{noise_mask_Folder}mask_{i}.png'
    noise_mask = cv2.imread(noise_mask_path)
    noise_indices = get_noise_index(noise_mask)
    original_image = cv2.imread(f'{original_Folder}im_{i}.png')
    noise_image = cv2.imread(f'{noise_Folder}im_{i}.png')
    # pad noise with reflection
    pad = kernel_size // 2  
    padded_noise = cv2.copyMakeBorder(noise_image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    for (x, y) in noise_indices:
        x_start = x + pad - kernel_size // 2
        x_end = x + pad + kernel_size // 2 + 1
        y_start = y + pad - kernel_size // 2
        y_end = y + pad + kernel_size // 2 + 1
        trainX = padded_noise[x_start:x_end, y_start:y_end, :]
        # H, W, C to H * W, C
        trainX = trainX.reshape(-1, 3)
        # Compute the median value for each channel
        trainY = original_image[x, y, :]
        assert trainX.shape[0] == kernel_size * kernel_size, "trainX shape mismatch"
        assert trainY.shape[0] == 3, "trainY shape mismatch"
        if i < 81:
            np.save(f'{outputTrainX}pixel_{i}_{x}_{y}.npy', trainX)
            np.save(f'{outputTrainY}label_{i}_{x}_{y}.npy', trainY)
        else:
            np.save(f'{outputTestX}pixel_{i}_{x}_{y}.npy', trainX)
            np.save(f'{outputTestY}label_{i}_{x}_{y}.npy', trainY)
        print(np.mean(trainX, axis=0), trainY)
