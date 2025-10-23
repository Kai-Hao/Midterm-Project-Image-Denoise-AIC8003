import os
import cv2
import torch
import numpy as np
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]
def unnormalize(image, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    """
    Unnormalize a numpy image with the given mean and std.
    """
    image = image * std + mean
    return np.clip(image, 0, 1)

def noise_mask_detection(image, kernel_size = 3):
    '''
    input: b, c, h, w tensor
    output: b, c, h, w tensor (0 for clean, 1 for noisy)
    '''
    b, c, h, w = image.size()
    device = image.device

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    image_unnorm = torch.clamp(image * std + mean, 0, 1)

    gray = torch.mean(image_unnorm, dim=1, keepdim=True)

    pad = kernel_size // 2
    padded_gray = torch.nn.functional.pad(gray, (pad, pad, pad, pad), mode='reflect')

    neighborhoods = torch.nn.functional.unfold(
        padded_gray, 
        kernel_size=kernel_size, 
        stride=1
    )  # b, k*k, h*w
    neighborhoods = neighborhoods.view(b, kernel_size*kernel_size, h, w)
    center_idx = kernel_size * kernel_size // 2
    center_pixels = neighborhoods[:, center_idx:center_idx+1, :, :]  # b, 1, h, w

    neighbor_mask = torch.ones(kernel_size*kernel_size, dtype=torch.bool, device=device)
    neighbor_mask[center_idx] = False
    neighbors = neighborhoods[:, neighbor_mask, :, :]  # b, (k*k-1), h, w
    diff = torch.abs(neighbors - center_pixels)  # b, (k*k-1), h, w
    threshold = 0.275
    large_diff_count = torch.sum(diff > threshold, dim=1, keepdim=True)  # b, 1, h, w
    total_neighbors = kernel_size * kernel_size - 1
    noise_ratio_threshold = 0.8
    noise_mask = (large_diff_count >= total_neighbors * noise_ratio_threshold).float()
    return noise_mask


def get_noise_index(noise_mask):
    '''
    ndarray: h, w, 1 # 0 for clean, 255 for noisy
    return: list of (x, y) coordinates of noisy pixels
    '''
    noise_positions = np.where(noise_mask[:, :, 0] == 255)
    noise_indices = list(zip(noise_positions[0], noise_positions[1]))
    return noise_indices

if __name__ == "__main__":
    os.makedirs('./Visualize/NoiseMask/', exist_ok=True)
    for i in range(1, 101, 1):
        if i == 82:
            continue
        image = cv2.imread(f'./Midterm100/noisy/im_{i}.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = torch.tensor(image)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = image.permute(2, 0, 1).unsqueeze(0)  # 1, c, h, w
        mask = noise_mask_detection(image, kernel_size=3)
        mask = mask.squeeze(0).permute(1, 2, 0).numpy() * 255  # h, w, c
        cv2.imwrite(f'./Visualize/NoiseMask/mask_{i}.png', mask.astype(np.uint8))
    '''
    inputFolder = './Visualize/NoiseMask/'
    for i in range(1, 101, 1):
        if i == 82:
            continue
        image = cv2.imread(f'./Visualize/NoiseMask/mask_{i}.png')
        noise_indices = get_noise_index(image)
        print(image.shape)
        print(len(noise_indices))
    '''
    
