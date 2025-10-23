from ray import method
from datareader import PixelReader
from torch.utils.data import DataLoader
from model import PixelModel
import torch
import torch.nn as nn
from tqdm import tqdm
if __name__ == "__main__":
    testDataset = PixelReader(Mode='test')
    testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False)
    testCriterion = nn.L1Loss()
    kernel_size = 3
    Fmethod = 'mean'  # 'mean' or 'median'
    test_loss = 0.0
    for pixels, labels in tqdm(testDataloader):
        pixels = pixels.cuda()
        labels = labels.cuda()
        # move center pixel out
        center_idx = kernel_size * kernel_size // 2
        mask = torch.ones(kernel_size * kernel_size, dtype=torch.bool, device=pixels.device)
        mask[center_idx] = False
        clean_pixels = pixels[:, mask, :]
        if Fmethod == 'mean':
            outputs = torch.mean(clean_pixels, dim=1)
        elif Fmethod == 'median':
            median_result = torch.median(clean_pixels, dim=1)
            outputs = median_result.values

        loss = testCriterion(outputs, labels)
        test_loss += loss.item() * pixels.size(0)
    avg_test_loss = float(test_loss) / len(testDataset)
    print(f"Test Loss: {avg_test_loss:.4f}")

