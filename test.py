import torch
from torchvision import transforms, datasets

from model import BrainTumorModel
from config import DEVICE

def test_model(model_path='best_model.pth'):
    # Load the model
    model = BrainTumorModel(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(DEVICE)
    model.eval()

    # Load the test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    test_dataset = datasets.ImageFolder(root="datasets/Testing", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total:.2f}%')    


if __name__ == '__main__':
    test_model()