import torch
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 0.001
EPOCHS = 25
