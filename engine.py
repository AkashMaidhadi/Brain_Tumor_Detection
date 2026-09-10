import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device):
    
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(data_loader): # tqdm -> to track the no.of batches
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = torch.nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)    
def valid_fn(data_loader, model, device):

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(data_loader): # tqdm -> to track the no.of batches
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item()

    return total_loss / len(data_loader)