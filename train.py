import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import BATCH_SIZE, DEVICE, LR, EPOCHS
from model import BrainTumorModel
from engine import train_fn, valid_fn

import sys
# Add the current directory to sys.path to find local modules
sys.path.append('.') # Or the appropriate path to your project root if running from a subdirectory

def main():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_df = datasets.ImageFolder(root = "datasets/Training", transform = transform)
    test_df = datasets.ImageFolder(root = "datasets/Testing", transform = transform)

    trainloader = DataLoader(train_df, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_df, batch_size=BATCH_SIZE, shuffle=False)

    model = BrainTumorModel(num_classes=4)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_valid_loss = np.inf
    for i in range(EPOCHS):
        train_loss = train_fn(trainloader, model, optimizer, DEVICE)
        valid_loss = valid_fn(testloader, model, DEVICE)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), "best_model.pth")
            print('SAVED MODEL')

        
        print(f'Epoch:{i+1} Train_loss: {train_loss:.4f}   Valid_loss: {valid_loss:.4f}')

if __name__ == '__main__':
    main()