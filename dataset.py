import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import cv2

class CervicalDataset(Dataset):
    def __init__(self, items, img_dir, image_size=(256, 256), transform=None, num_landmarks=4):
        self.items = items
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        self.num_landmarks = num_landmarks

    def __len__(self):
        return len(self.items)

    def _generate_heatmap(self, points, height, width, sigma=4):
        heatmaps = np.zeros((self.num_landmarks, height, width), dtype=np.float32)
        for i, (x, y) in enumerate(points):
            if x < 0 or y < 0:
                continue
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        return heatmaps

    def __getitem__(self, idx):
        item = self.items[idx]
        image_name = item['id']
        img_path = os.path.join(self.img_dir, image_name + '.png')
        image = Image.open(img_path).convert("L")
        image = np.array(image)

        h_ori, w_ori = image.shape
        h, w = self.image_size
        
        keypoints = [(-1, -1)] * self.num_landmarks
        for ann in item['annotations']:
            if ann['type'] != 'points':
                continue
            lid = ann['label_id']
            x, y = ann['points']
            if lid == 1:
                keypoints[0] = (x * w / w_ori, y * h / h_ori)
            elif lid == 2:
                keypoints[1] = (x * w / w_ori, y * h / h_ori)
            elif lid == 21:
                keypoints[2] = (x * w / w_ori, y * h / h_ori)
            elif lid == 22:
                keypoints[3] = (x * w / w_ori, y * h / h_ori)

        image = cv2.resize(image, (w, h))

        heatmaps = self._generate_heatmap(keypoints, h, w)

        conf_map = np.zeros((2, h, w), dtype=np.float32)
        if all([kp[0] >= 0 for kp in keypoints]):
            line1 = cv2.line(np.zeros((h, w), dtype=np.uint8),
                            tuple(np.round(keypoints[0]).astype(int)),
                            tuple(np.round(keypoints[1]).astype(int)), 1, 1)
            line2 = cv2.line(np.zeros((h, w), dtype=np.uint8),
                            tuple(np.round(keypoints[2]).astype(int)),
                            tuple(np.round(keypoints[3]).astype(int)), 1, 1)
            conf_map[0] = line1
            conf_map[1] = line2

        target = np.concatenate([heatmaps, conf_map], axis=0)

        if self.transform:
            augmented = self.transform(image=image, mask=target.transpose(1, 2, 0))
            image = augmented['image']
            target = augmented['mask'].permute(2, 0, 1)

        return image, target
        