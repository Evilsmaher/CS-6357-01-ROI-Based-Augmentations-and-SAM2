import argparse
import torch
import os
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms

def load_model(device):
    # Load the segmentation model
    model = smp.Unet(encoder_name='resnet34',
                     encoder_weights='imagenet',
                     in_channels=3,
                     classes=1,
                     decoder_attention_type='scse')
    
    # Load pretrained weights
    model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'unet_residual_prediction.pth'), map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image_path, device):
    # Load and preprocess the image
    input_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return input_tensor, input_image

def segment_image(model, image_tensor):
    with torch.no_grad():
        # Apply the model and sigmoid activation
        output = model(image_tensor)
        output = torch.sigmoid(output)  # Sigmoid for binary segmentation
        return output

def create_binary_mask(output_tensor, threshold=0.5):
    # Create a binary mask by thresholding the model output
    mask = (output_tensor > threshold).float()
    return mask

def save_binary_mask(mask, original_image_path, original_image):
    # Convert the mask to a binary image and save it
    mask = mask.squeeze().cpu().numpy()  # Remove batch dimension and move to CPU
    mask = (mask * 255).astype(np.uint8)  # Scale to [0, 255] for saving as an image
    mask_image = Image.fromarray(mask).resize(original_image.size, Image.BILINEAR)
    
    # Save the binary mask
    mask_filename = f"mask_{os.path.basename(original_image_path)}"
    mask_filepath = os.path.join(os.path.dirname(original_image_path), mask_filename)
    mask_image.save(mask_filepath)

def main():
    parser = argparse.ArgumentParser(description="Generate and save a binary segmentation mask for an input image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the model
    model = load_model(device)
    
    # Preprocess the input image
    input_tensor, original_image = preprocess_image(args.image_path, device)
    
    # Segment the image
    output = segment_image(model, input_tensor)
    
    # Generate a binary mask
    binary_mask = create_binary_mask(output)
    
    # Save the binary mask to the same folder as the original image
    save_binary_mask(binary_mask, args.image_path, original_image)

if __name__ == "__main__":
    main()
