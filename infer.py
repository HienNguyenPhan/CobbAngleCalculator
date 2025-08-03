import sys
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from utils import extract_keypoints_from_heatmap, predict_cobb_from_image
from model import get_model
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def infer(IMAGE_DIR, CHECKPOINT_DIR):
    model = get_model()
    model.to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT_DIR))
    print("Load state successful")
    cobb_angle, keypoints = predict_cobb_from_image(IMAGE_DIR, model, device=device)
    return cobb_angle

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("The cobb angle is: <angle>")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    checkpoint_dir = sys.argv[2]
    angle = infer(image_dir, checkpoint_dir)
