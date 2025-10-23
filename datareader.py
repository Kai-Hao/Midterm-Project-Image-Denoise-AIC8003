import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as TF
from einops import rearrange
from utils import get_noise_index

class ImageReader(Dataset):
    def __init__(self, DataDir = './Midterm100/', Mode = 'train', ColorMode = cv2.COLOR_BGR2RGB):
        self.mode = Mode
        self.colormode = ColorMode
        if Mode == 'train':
            self.X = [os.path.join(DataDir, 'noisy', f"im_{i}.png") for i in range(1, 81)]
            self.Y = [os.path.join(DataDir, 'original', f"im_{i}.png") for i in range(1, 81)]
            self.TF = TF.Compose([
                TF.ToPILImage(),
                TF.Resize((128, 128)),
                TF.ToTensor(),
                TF.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        else:
            self.X = [os.path.join(DataDir, 'noisy', f"im_{i}.png") for i in range(81, 101) if i != 82]
            self.Y = [os.path.join(DataDir, 'original', f"im_{i}.png") for i in range(81, 101) if i != 82]
            self.TF = TF.Compose([
                TF.ToPILImage(),
                TF.Resize((128, 128)),
                TF.ToTensor(),
                TF.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        noisy_image = cv2.imread(self.X[idx])
        original_image = cv2.imread(self.Y[idx])
        # Apply Transformations
        noisy_image = cv2.cvtColor(noisy_image, self.colormode)
        original_image = cv2.cvtColor(original_image, self.colormode)
        if self.mode == 'train':
            noisy_image = self.TF(noisy_image)
            original_image = self.TF(original_image)
            return noisy_image, original_image
        else:
            noisy_image = self.TF(noisy_image)
            return noisy_image, original_image

from model import PixelModel

class ImageReaderWithNoiseMap(Dataset):
    def __init__(self, DataDir = './Midterm100/', Mode = 'train', ColorMode = cv2.COLOR_BGR2RGB):
        self.mode = Mode
        self.colormode = ColorMode
        self.model = PixelModel(3 * 3 * 3)
        self.model.load_state_dict(torch.load('./PixelModel/best_pixel_model_best.pth'))
        self.model.eval()
        self.model = self.model
        if Mode == 'train':
            self.X = [os.path.join(DataDir, 'noisy', f"im_{i}.png") for i in range(1, 81)]
            self.Y = [os.path.join(DataDir, 'original', f"im_{i}.png") for i in range(1, 81)]
            self.X_noise = [os.path.join(DataDir, 'noiseMask', f"mask_{i}.png") for i in range(1, 81)]
            self.TF = TF.Compose([
                TF.ToPILImage(),
                TF.Resize((128, 128)),
                TF.ToTensor(),
                TF.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
        else:
            self.X = [os.path.join(DataDir, 'noisy', f"im_{i}.png") for i in range(81, 101) if i != 82]
            self.Y = [os.path.join(DataDir, 'original', f"im_{i}.png") for i in range(81, 101) if i != 82]
            self.X_noise = [os.path.join(DataDir, 'noiseMask', f"mask_{i}.png") for i in range(81, 101) if i != 82]
            self.TF = TF.Compose([
                TF.ToPILImage(),
                TF.Resize((128, 128)),
                TF.ToTensor(),
                TF.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        noisy_image = cv2.imread(self.X[idx])
        original_image = cv2.imread(self.Y[idx])
        noise_map = cv2.imread(self.X_noise[idx])
        noise_indices = get_noise_index(noise_map)
        kernel_size = 3
        pad = kernel_size // 2  
        padded_noise = cv2.copyMakeBorder(noisy_image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        for (x, y) in noise_indices:
            x_start = x + pad - kernel_size // 2
            x_end = x + pad + kernel_size // 2 + 1
            y_start = y + pad - kernel_size // 2
            y_end = y + pad + kernel_size // 2 + 1
            inputX = padded_noise[x_start:x_end, y_start:y_end, :]
            inputX = inputX.reshape(-1, 3)
            inputX = torch.tensor(inputX, dtype=torch.float32).unsqueeze(0)  # 1, k*k, 3
            inputX = inputX / 255.0
            with torch.no_grad():
                outputY = self.model(inputX)  # 1, 3
            noisy_image[x, y, :] = (outputY.squeeze().numpy() * 255.0).astype(np.uint8)
        # Apply Transformations
        noisy_image = cv2.cvtColor(noisy_image, self.colormode)
        original_image = cv2.cvtColor(original_image, self.colormode)
        if self.mode == 'train':
            noisy_image = self.TF(noisy_image)
            original_image = self.TF(original_image)
            return noisy_image, original_image
        else:
            noisy_image = self.TF(noisy_image)
            return noisy_image, original_image



class PixelReader(Dataset):
    def __init__(self, Mode = 'train'):
        self.mode = Mode
        if Mode == 'train':
            self.inputFolder = './Midterm100/pixeltrain/pixel/'
            self.labelFolder = './Midterm100/pixeltrain/label/'
            self.trainX = [os.path.join(self.inputFolder, f) for f in os.listdir(self.inputFolder) if f.endswith('.npy')]
            self.trainY = [os.path.join(self.labelFolder, f) for f in os.listdir(self.labelFolder) if f.endswith('.npy')]
        else:
            self.inputFolder = './Midterm100/pixeltest/pixel/'
            self.labelFolder = './Midterm100/pixeltest/label/'
            self.testX = [os.path.join(self.inputFolder, f) for f in os.listdir(self.inputFolder) if f.endswith('.npy')]
            self.testY = [os.path.join(self.labelFolder, f) for f in os.listdir(self.labelFolder) if f.endswith('.npy')]
    def __len__(self):
        if self.mode == 'train':
            return len(self.trainX)
        else:
            return len(self.testX)
    def __getitem__(self, idx):
        if self.mode == 'train':
            pixel_data = np.load(self.trainX[idx])  # k*k, 3
            label_data = np.load(self.trainY[idx])  # 3,
        else:
            pixel_data = np.load(self.testX[idx])  # k*k, 3
            label_data = np.load(self.testY[idx])  # 3,
        pixel_data = torch.tensor(pixel_data, dtype=torch.float32)
        label_data = torch.tensor(label_data, dtype=torch.float32)
        return pixel_data, label_data
    
# TEST FUNCTION
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils import unnormalize
    '''
    dataset = ImageReader(Mode='train')
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)
    for i, (noisy, original) in enumerate(dataloader):
        original = rearrange(original, 'b c h w -> b h w c').numpy()
        noisy = rearrange(noisy, 'b c h w -> b h w c').numpy()
        break
    plt.figure(figsize=(12, 6))
    # subplot 8 x 4
    for i in range(4):
        for j in range(4):
            plt.subplot(4, 8, i * 8 + j * 2 + 1)
            plt.imshow((unnormalize(noisy[i * 4 + j]) * 255).astype(np.uint8))
            plt.title("Noisy Image")
            plt.axis("off")

            plt.subplot(4, 8, i * 8 + j * 2 + 2)
            plt.imshow((unnormalize(original[i * 4 + j]) * 255).astype(np.uint8))
            plt.title("Original Image")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
    '''
    dataset = ImageReaderWithNoiseMap(Mode='train')
    dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)
    for i, (noisy, original) in enumerate(dataloader):
        original = rearrange(original, 'b c h w -> b h w c').numpy()
        noisy = rearrange(noisy, 'b c h w -> b h w c').numpy()
        # show original images and noise-removed images side by side
        plt.figure(figsize=(6, 6))
        for i in range(4):
            for j in range(4):
                plt.subplot(4, 8, i * 8 + j * 2 + 1)
                plt.imshow((unnormalize(noisy[i * 4 + j]) * 255).astype(np.uint8))
                plt.title("Noisy Image")
                plt.axis("off")

                plt.subplot(4, 8, i * 8 + j * 2 + 2)
                plt.imshow((unnormalize(original[i * 4 + j]) * 255).astype(np.uint8))
                plt.title("Original Image")
                plt.axis("off")

        plt.tight_layout()
        plt.show()