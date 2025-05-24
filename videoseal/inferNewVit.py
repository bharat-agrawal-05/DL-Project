import argparse
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from newVit import VisionTransformerBinary

def predict_binary_string(model, image_path, image_size=256, threshold=0.5):
    """
    Predict binary string from image
    
    Args:
        model: Trained model
        image_path: Path to input image
        image_size: Input image size
        threshold: Threshold for binary prediction
        
    Returns:
        Predicted binary string
    """
    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    # Move to the same device as model
    device = next(model.parameters()).device
    image = image.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image)
    
    # Convert to binary
    binary_pred = (output >= threshold).int().squeeze().cpu().numpy()
    binary_string = ''.join(map(str, binary_pred))
    
    # Also return raw probabilities
    probabilities = output.squeeze().cpu().numpy()
    
    return binary_string, probabilities

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Get binary string length from the model
    binary_length = checkpoint['model_state_dict']['head.1.weight'].shape[0]
    
    # Initialize model
    model = VisionTransformerBinary(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        emb_dim=args.emb_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=0.0,  # Set to 0 for inference
        binary_length=binary_length
    ).to(device)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Predict binary string
    binary_string, probabilities = predict_binary_string(
        model, 
        args.image_path, 
        args.image_size, 
        args.threshold
    )
    
    print(f"\nPredicted binary string: {binary_string}")
    print("Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  Bit {i}: {prob:.4f} -> {int(prob >= args.threshold)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with Vision Transformer binary predictor")
    
    # Model parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--emb_dim", type=int, default=768, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=12, help="Transformer depth")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--mlp_ratio", type=float, default=4.0, help="MLP ratio")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary prediction")
    
    args = parser.parse_args()
    main(args)