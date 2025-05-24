import torch
from torchvision import transforms
from PIL import Image
import os

from vit_regressor import VisionTransformerRegressor  # Make sure this matches your model file name

# Set paths
checkpoint_path = "vit_regressor_checkpoint.pth"
image_path = "path/to/image.jpg"  # ← Update this with the image you want to test

# Preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformerRegressor().to(device)

checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess image
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0).to(device)  # Add batch dim

# Predict
with torch.no_grad():
    prediction = model(image)

print("Predicted vector (dim=128):")
print(prediction.squeeze().cpu().numpy())
