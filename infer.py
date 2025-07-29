import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from utils import extract_keypoints_from_heatmap, predict_cobb_from_image
from model import Model
import matplotlib as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def infer(IMAGE_DIR, CHECKPOINT_DIR):
    model = Model()
    model.cuda().eval()
    model.load_state_dict(torch.load(CHECKPOINT_DIR))
    cobb_angle, keypoints = predict_cobb_from_image(IMAGE_DIR, model, device=device)
    print("Predicted Cobb Angle:", cobb_angle)
