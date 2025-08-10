import sys
import torch
from utils import predict_cobb_from_image
from model import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_dir = "./best_model.pth"

def infer(IMAGE_DIR, CHECKPOINT_DIR):
    model = Model()
    model.to(device).eval()
    model.load_state_dict(torch.load(CHECKPOINT_DIR))
    print("Load state successful")
    cobb_angle, _ = predict_cobb_from_image(IMAGE_DIR, model, device=device)
    return cobb_angle


if len(sys.argv) != 2:
    print("Usage: python infer.py <image_path>")
    sys.exit(1)

image_dir = sys.argv[1]
angle = infer(image_dir, checkpoint_dir)
print(f"The Cobb angle is: {angle:.2f}")
