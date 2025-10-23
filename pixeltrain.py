from datareader import PixelReader
from torch.utils.data import DataLoader
from model import PixelModel
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
if __name__ == "__main__":
    end_learning_rate = 1e-5
    trainDataset = PixelReader(Mode='train')
    testDataset = PixelReader(Mode='test')
    trainDataloader = DataLoader(trainDataset, batch_size=32, shuffle=True)
    testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False)
    model = PixelModel(input_size = 3*3*3, hidden_size = 128).cuda()
    criterion = nn.MSELoss()
    testcriterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=70, T_mult=2, eta_min=end_learning_rate)
    num_epochs = 70
    min_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (pixels, labels) in enumerate(tqdm(trainDataloader)):
            pixels = pixels.cuda()  # (B, k*k, 3)
            labels = labels.cuda()  # (B, 3)
            pixels = pixels/255.0  # Normalize to [0, 1]
            labels = labels/255.0  # Normalize to [0, 1]
            optimizer.zero_grad()
            outputs = model(pixels)  # (B, 3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * pixels.size(0)
        
        avg_loss = running_loss / len(trainDataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        scheduler.step()
        # Evaluation on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for pixels, labels in tqdm(testDataloader):
                pixels = pixels.cuda()
                labels = labels.cuda()
                pixels = pixels/255.0
                labels = labels/255.0
                outputs = model(pixels)
                outputs = (outputs*255).int().float()
                labels = (labels*255).int().float()
                loss = testcriterion(outputs, labels)
                test_loss += loss.item() * pixels.size(0)
                
        avg_test_loss = float(test_loss) / len(testDataset)
        print(f"Test Loss after Epoch {epoch+1}: {avg_test_loss:.4f}")

        # Save the model checkpoint if it has the lowest test loss
        if avg_test_loss < min_loss:
            min_loss = avg_test_loss
            torch.save(model.state_dict(), './PixelModel/best_pixel_model.pth')
            print(f"Model saved at epoch {epoch+1} with test loss {min_loss:.4f}")
    print("Training complete. Best Test Loss: {:.4f}".format(min_loss))