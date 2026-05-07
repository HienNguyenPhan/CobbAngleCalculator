import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image


class CervicalDataset(Dataset):
    def __init__(
        self,
        items,
        img_dir,
        image_size=(256, 256),
        transform=None,
        include_masks=True,
        sigma=4.0
    ):
        self.items = items
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transform
        self.include_masks = include_masks
        self.sigma = sigma

        # Keypoint names we care about for Cobb angle (C2 inferior + C7 inferior)
        self.keypoint_names = [
            "C2 bottom left",
            "C2 bottom right",
            "C7 bottom left",
            "C7 bottom right"
        ]
        self.num_landmarks = len(self.keypoint_names)

        # Vertebra polygon label names -> class id (starting from 1)
        self.vertebra_names = ["C2", "C3", "C4", "C5", "C6", "C7"]
        self.vertebra_to_class = {name: i + 1 for i, name in enumerate(self.vertebra_names)}

    def __len__(self):
        return len(self.items)

    def _generate_heatmap(self, points, height, width):
        """Generate Gaussian heatmaps for the 4 Cobb keypoints"""
        heatmaps = np.zeros((self.num_landmarks, height, width), dtype=np.float32)
        for i, (x, y) in enumerate(points):
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
        return heatmaps

    def _generate_mask(self, polygons_dict, height, width):
        """
        Generate multi-class segmentation mask.
        Returns: (H, W) int64 where value = class_id (0=bg, 1=C2, ..., 6=C7).
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        for label_name, points_flat in polygons_dict.items():
            if label_name not in self.vertebra_to_class:
                continue
            class_id = self.vertebra_to_class[label_name]

            pts = np.array(points_flat).reshape(-1, 2).astype(np.int32)
            if len(pts) < 3:
                continue

            cv2.fillPoly(mask, [pts], int(class_id))

        return mask.astype(np.int64)

    def __getitem__(self, idx):
        item = self.items[idx]
        image_name = item["id"]
        img_path = os.path.join(self.img_dir, f"{image_name}.png")

        # Load and convert to grayscale
        image = np.array(Image.open(img_path).convert("L"))
        h_ori, w_ori = image.shape
        h, w = self.image_size

        # Extract keypoints for heatmaps
        keypoints = [(-1.0, -1.0)] * self.num_landmarks

        for ann in item.get("annotations", []):
            if ann.get("type") != "points":
                continue
            lid = ann.get("label_id")
            if lid is None or len(ann.get("points", [])) != 2:
                continue
            x, y = ann["points"]
            if lid == 1:
                keypoints[0] = (x * w / w_ori, y * h / h_ori)
            elif lid == 2:
                keypoints[1] = (x * w / w_ori, y * h / h_ori)
            elif lid == 21:
                keypoints[2] = (x * w / w_ori, y * h / h_ori)
            elif lid == 22:
                keypoints[3] = (x * w / w_ori, y * h / h_ori)

        # Extract polygons for masks
        polygons_dict = {}
        if self.include_masks:
            for ann in item.get("annotations", []):
                if ann.get("type") != "polygon":
                    continue
                lid = ann.get("label_id")
                if lid is None:
                    continue

                if lid == 24:
                    name = "C2"
                elif lid == 25:
                    name = "C3"
                elif lid == 26:
                    name = "C4"
                elif lid == 27:
                    name = "C5"
                elif lid == 28:
                    name = "C6"
                elif lid == 29:
                    name = "C7"
                else:
                    continue

                points = ann.get("points", [])
                scaled = []
                for i in range(0, len(points), 2):
                    x_scaled = points[i] * w / w_ori
                    y_scaled = points[i + 1] * h / h_ori
                    scaled.extend([x_scaled, y_scaled])
                polygons_dict[name] = scaled

        # Resize image
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

        # Generate targets
        heatmaps = self._generate_heatmap(keypoints, h, w)

        if self.include_masks and polygons_dict:
            seg_mask = self._generate_mask(polygons_dict, h, w)
            seg_mask = np.expand_dims(seg_mask, axis=0)            # (1, H, W) int64 class ids
            target = np.concatenate([heatmaps, seg_mask.astype(np.float32)], axis=0)
        else:
            seg_mask = np.zeros((1, h, w), dtype=np.int64)
            target = np.concatenate([heatmaps, seg_mask.astype(np.float32)], axis=0)

        # Apply transforms
        if self.transform:
            augmented = self.transform(
                image=image,
                mask=target.transpose(1, 2, 0)
            )
            image = augmented["image"]
            target = augmented["mask"].permute(2, 0, 1) if hasattr(augmented["mask"], "permute") else augmented["mask"].transpose(2, 0, 1)

        # Convert image to tensor-like array if needed
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        return image, target