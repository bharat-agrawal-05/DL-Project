from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np

def calculate_ssim(imageA, imageB):
    # Resize imageB to match dimensions of imageA
    imageB = imageB.resize(imageA.size)
    # Convert PIL images to numpy arrays
    imageA_array = np.array(imageA.convert('L'))  # Convert to grayscale
    imageB_array = np.array(imageB.convert('L'))  # Convert to grayscale
    # Compute SSIM between two images
    ssim_value, _ = ssim(imageA_array, imageB_array, full=True)
    return ssim_value

if __name__ == "__main__":
    imageA = Image.open("test/sa_1.jpg")
    imageB = Image.open("out.jpg")
    ssim_result = calculate_ssim(imageA, imageB)
    print(f"SSIM between test/sa_1.jpg and out.jpg: {ssim_result}")