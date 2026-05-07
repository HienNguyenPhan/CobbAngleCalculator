import sys
import torch
import cv2
import numpy as np
from PIL import Image
from model import CervicalMultiTaskTransformer
from utils import extract_landmarks_from_heatmaps_weighted, calculate_cobb_angle

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_checkpoint(checkpoint_path, model, device='cpu'):
    """Load a saved checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model weights loaded")
    
    return checkpoint


def infer_cobb_angle(image_path, model, device='cuda', image_size=(256, 256)):
    """
    Infer Cobb angle from a single image
    
    Args:
        image_path: Path to the image file
        model: Trained model
        device: Device to run on
        image_size: Image size for model input
    
    Returns:
        Dictionary with predictions and metrics
    """
    # Load image
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    orig_shape = image_np.shape
    
    # Resize image
    image_resized = cv2.resize(image_np, image_size)
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_resized).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # Normalize
    image_tensor = (image_tensor - 0.485) / 0.229
    
    # Inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
    
    # Extract heatmaps and segmentation
    pred_heatmaps = output[0, :4].unsqueeze(0)
    pred_seg = torch.softmax(output[0, 4:11], dim=0)
    
    # Extract landmarks
    landmarks = extract_landmarks_from_heatmaps_weighted(pred_heatmaps)[0]
    
    # Calculate Cobb angle
    if all(pt is not None for pt in landmarks):
        cobb_angle = calculate_cobb_angle(landmarks)
    else:
        cobb_angle = 0.0
        print("⚠️  Warning: Could not detect all landmarks")
    
    return {
        'cobb_angle': cobb_angle,
        'landmarks': landmarks,
        'seg_map': pred_seg.cpu().numpy(),
        'heatmaps': pred_heatmaps.squeeze(0).cpu().numpy()
    }


def visualize_inference(image_path, results, image_size=(256, 256)):
    """Visualize inference results"""
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, image_size)
    
    # Convert to BGR for color visualization
    image_color = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2BGR)
    
    # Draw landmarks
    landmarks = results['landmarks']
    for i, pt in enumerate(landmarks):
        if pt is not None:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(image_color, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image_color, f"{i}", (x+10, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw lines connecting landmarks for Cobb angle visualization
    if all(pt is not None for pt in landmarks):
        p0 = tuple(np.int32(landmarks[0]))
        p1 = tuple(np.int32(landmarks[1]))
        p2 = tuple(np.int32(landmarks[2]))
        p3 = tuple(np.int32(landmarks[3]))
        
        cv2.line(image_color, p0, p1, (255, 255, 0), 2)  # Yellow line
        cv2.line(image_color, p2, p3, (0, 255, 255), 2)  # Cyan line
    
    # Add text
    cv2.putText(image_color, f"Cobb Angle: {results['cobb_angle']:.2f}°", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image_color


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python infer.py <image_path> <checkpoint_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    
    # Load model
    model = CervicalMultiTaskTransformer(encoder_name='mit_b2', encoder_weights=None)
    model = model.to(device)
    
    # Load checkpoint
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    
    # Run inference
    print(f"\nInferring on image: {image_path}")
    results = infer_cobb_angle(image_path, model, device=device)
    
    # Print results
    print(f"\n{'='*50}")
    print("INFERENCE RESULTS")
    print(f"{'='*50}")
    print(f"Cobb Angle: {results['cobb_angle']:.2f}°")
    print(f"Landmarks:")
    for i, pt in enumerate(results['landmarks']):
        if pt is not None:
            print(f"  L{i}: ({pt[0]:.1f}, {pt[1]:.1f})")
        else:
            print(f"  L{i}: Not detected")
    print(f"{'='*50}")
