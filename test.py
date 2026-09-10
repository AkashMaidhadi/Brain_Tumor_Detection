
import torch
from PIL import Image
from torchvision import transforms
from model import BrainTumorModel

# 1. load model
model = BrainTumorModel(num_classes=4)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.eval()

# 2. define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 3. class names in the same order as ImageFolder folders
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# 4. load one image
img = Image.open("path/to/your/image.jpg").convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # add batch dimension

# 5. predict
with torch.no_grad():
    output = model(img_tensor)
    pred_idx = torch.argmax(output, dim=1).item()

# 6. print class label
print("Predicted class:", class_names[pred_idx])

