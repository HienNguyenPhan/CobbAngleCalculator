import numpy as np
import cv2
import itertools
import torch
import matplotlib as plt
from PIL import Image
from torchvision import transforms
from scipy.ndimage import center_of_mass

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,], std=[0.229,])
])

def get_max_location(heatmap):
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return np.array([x, y], dtype=np.float32)

def get_keypoints_from_heatmap(heatmaps):
    return [get_max_location(heatmap) for heatmap in heatmaps]

def compute_nme(pred_keypoints, gt_keypoints):
    pred_keypoints = np.array(pred_keypoints)
    gt_keypoints = np.array(gt_keypoints)

    error = np.linalg.norm(pred_keypoints - gt_keypoints, axis=1)
    norm = np.linalg.norm(gt_keypoints[0] - gt_keypoints[-1]) + 1e-6
    return np.mean(error) / norm

def extract_keypoints_from_heatmap(heatmap):
    all_candidates = heatmap_to_candidates(heatmap)
    keypoints = []
    best_scores = []

    for i in range(2):
        c1s, c2s = all_candidates[2 * i], all_candidates[2 * i + 1]
        if not c1s or not c2s:
            keypoints.extend([(0, 0), (0, 0)])
            best_scores.append(0.0)
            continue

        scores = []
        pairs = []
        conf_map = heatmap[4 + i]
        for c1, c2 in itertools.product(c1s, c2s):
            line = np.zeros_like(conf_map)
            p1 = tuple(np.round(c1).astype(int))
            p2 = tuple(np.round(c2).astype(int))
            cv2.line(line, p1, p2, 1, 3)
            score = (conf_map * line).sum() / (line.sum() + 1e-6)
            scores.append(score)
            pairs.append((c1, c2))
        best = pairs[np.argmax(scores)]
        best_scores.append(np.max(scores))
        keypoints.extend(best)
    return keypoints, best_scores

def heatmap_to_candidates(heatmap):
    all_candidates = []
    for ch in heatmap[:4]:
        max_val = ch.max()
        if max_val <= 0:
            raise RuntimeError('Empty heatmap')
        thresh = max_val / 2
        labels = (ch >= thresh).astype(np.uint8)
        labeled = cv2.connectedComponentsWithStats(labels, 8, cv2.CV_32S)[1]
        candidates = []
        for i in range(1, labeled.max() + 1):
            mask = (labeled == i)
            center = center_of_mass(ch * mask)
            candidates.append(center[::-1])  # (x, y)
        all_candidates.append(candidates)
    return all_candidates

def predict_cobb_from_image(image_path, model, device='cuda', image_size=(256, 256), visualize=True):
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    orig_image = image.copy()
    
    image_resized = cv2.resize(image, image_size)
    
    image_tensor = transform(Image.fromarray(image_resized)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        heatmap = output.squeeze(0).cpu().numpy()

    keypoints, _ = extract_keypoints_from_heatmap(heatmap)

    def angle_between(p1, p2):
        delta = np.array(p2) - np.array(p1)
        angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi
        return angle
    
    angle1 = angle_between(keypoints[0], keypoints[1])
    angle2 = angle_between(keypoints[2], keypoints[3])
    cobb_angle = abs(angle1 - angle2)
    
    if visualize:
        image_color = cv2.cvtColor(cv2.resize(orig_image, image_size), cv2.COLOR_GRAY2BGR)
        for (x, y) in keypoints:
            cv2.circle(image_color, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.line(image_color, tuple(np.int32(keypoints[0])), tuple(np.int32(keypoints[1])), (0, 255, 255), 2)
        cv2.line(image_color, tuple(np.int32(keypoints[2])), tuple(np.int32(keypoints[3])), (255, 0, 255), 2)
        cv2.putText(image_color, f"Cobb: {cobb_angle:.2f} deg", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        plt.imshow(image_color)
        plt.title("Predicted Cobb Angle")
        plt.axis("off")
        plt.show()

    return cobb_angle, keypoints

