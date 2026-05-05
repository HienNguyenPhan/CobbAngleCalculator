import numpy as np
import cv2
import itertools
import torch
import matplotlib.pyplot as plt
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


# ============================================================================
# SUBPIXEL REFINEMENT METHODS (More Accurate than Argmax)
# ============================================================================

def get_max_location_gaussian_weighted(heatmap, window=5):
    """
    Method: Gaussian-weighted centroid
    Better than argmax - gives subpixel accuracy
    """
    y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    h, w = heatmap.shape
    half_win = window // 2
    
    # Extract local window around peak
    x_start = max(0, x_max - half_win)
    x_end = min(w, x_max + half_win + 1)
    y_start = max(0, y_max - half_win)
    y_end = min(h, y_max + half_win + 1)
    
    local_patch = heatmap[y_start:y_end, x_start:x_end]
    
    # Create coordinate grids
    x_coords = np.arange(x_start, x_end)
    y_coords = np.arange(y_start, y_end)
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Weight by heatmap values (squared to emphasize peak)
    weights = local_patch ** 2
    weights = weights / (weights.sum() + 1e-7)
    
    # Compute weighted centroid (subpixel coordinates)
    x_refined = (xx * weights).sum()
    y_refined = (yy * weights).sum()
    
    return np.array([x_refined, y_refined], dtype=np.float32)


def get_max_location_parabolic(heatmap):
    """
    Method: Parabolic peak fitting
    Classic subpixel refinement - fits parabola to peak
    """
    y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    h, w = heatmap.shape
    
    # Need at least 1 pixel margin
    if x_max == 0 or x_max == w-1 or y_max == 0 or y_max == h-1:
        return np.array([x_max, y_max], dtype=np.float32)
    
    # Get 3x3 neighborhood
    c = heatmap[y_max, x_max]      # center
    l = heatmap[y_max, x_max - 1]  # left
    r = heatmap[y_max, x_max + 1]  # right
    t = heatmap[y_max - 1, x_max]  # top
    b = heatmap[y_max + 1, x_max]  # bottom
    
    # Parabolic fit in x direction
    if (l + r - 2*c) != 0:
        dx = 0.5 * (l - r) / (l + r - 2*c)
    else:
        dx = 0
    
    # Parabolic fit in y direction
    if (t + b - 2*c) != 0:
        dy = 0.5 * (t - b) / (t + b - 2*c)
    else:
        dy = 0
    
    # Clamp to reasonable range (max 0.5 pixel shift)
    dx = np.clip(dx, -0.5, 0.5)
    dy = np.clip(dy, -0.5, 0.5)
    
    return np.array([x_max + dx, y_max + dy], dtype=np.float32)


def get_max_location_taylor(heatmap):
    """
    Method: Taylor expansion with Hessian matrix
    Most accurate but computationally expensive
    """
    y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    h, w = heatmap.shape
    
    if x_max <= 1 or x_max >= w-2 or y_max <= 1 or y_max >= h-2:
        return np.array([x_max, y_max], dtype=np.float32)
    
    # Compute gradients
    dx = (heatmap[y_max, x_max+1] - heatmap[y_max, x_max-1]) / 2
    dy = (heatmap[y_max+1, x_max] - heatmap[y_max-1, x_max]) / 2
    
    # Compute Hessian (second derivatives)
    dxx = heatmap[y_max, x_max+1] - 2*heatmap[y_max, x_max] + heatmap[y_max, x_max-1]
    dyy = heatmap[y_max+1, x_max] - 2*heatmap[y_max, x_max] + heatmap[y_max-1, x_max]
    dxy = (heatmap[y_max+1, x_max+1] - heatmap[y_max+1, x_max-1] - 
           heatmap[y_max-1, x_max+1] + heatmap[y_max-1, x_max-1]) / 4
    
    # Hessian matrix
    H = np.array([[dxx, dxy], [dxy, dyy]])
    
    # Check if Hessian is invertible
    det = np.linalg.det(H)
    if abs(det) < 1e-6:
        return np.array([x_max, y_max], dtype=np.float32)
    
    # Compute offset: -H^(-1) * gradient
    gradient = np.array([dx, dy])
    try:
        offset = -np.linalg.solve(H, gradient)
        offset = np.clip(offset, -1, 1)  # Limit to 1 pixel shift
        return np.array([x_max + offset[0], y_max + offset[1]], dtype=np.float32)
    except:
        return np.array([x_max, y_max], dtype=np.float32)


def get_max_location_center_of_mass(heatmap, threshold_ratio=0.5):
    """
    Method: Center of mass of thresholded region
    Good for broader peaks
    """
    max_val = heatmap.max()
    if max_val <= 0:
        return np.array([0, 0], dtype=np.float32)
    
    # Threshold at percentage of peak
    threshold = max_val * threshold_ratio
    mask = heatmap >= threshold
    
    # Calculate center of mass
    y_cm, x_cm = center_of_mass(heatmap * mask)
    
    if np.isnan(x_cm) or np.isnan(y_cm):
        # Fallback to argmax
        y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return np.array([x_max, y_max], dtype=np.float32)
    
    return np.array([x_cm, y_cm], dtype=np.float32)


def get_keypoints_from_heatmap_refined(heatmaps, method='gaussian_weighted'):
    """
    Extract keypoints with subpixel refinement
    
    Args:
        heatmaps: list or array of heatmaps
        method: 'argmax', 'gaussian_weighted', 'parabolic', 'taylor', 'center_of_mass'
    
    Returns:
        list of (x, y) coordinates with subpixel accuracy
    """
    if method == 'argmax':
        return [get_max_location(hm) for hm in heatmaps]
    elif method == 'gaussian_weighted':
        return [get_max_location_gaussian_weighted(hm) for hm in heatmaps]
    elif method == 'parabolic':
        return [get_max_location_parabolic(hm) for hm in heatmaps]
    elif method == 'taylor':
        return [get_max_location_taylor(hm) for hm in heatmaps]
    elif method == 'center_of_mass':
        return [get_max_location_center_of_mass(hm) for hm in heatmaps]
    else:
        raise ValueError(f"Unknown method: {method}")


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


def calculate_cobb_angle(keypoints):
    """Compute Cobb angle from 4 keypoints (two lines: (0,1) and (2,3)).

    Args:
        keypoints: iterable of at least 4 (x, y) points.

    Returns:
        Cobb angle in degrees (float).
    """

    def angle_between(p1, p2):
        delta = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
        return float(np.arctan2(delta[1], delta[0]) * 180.0 / np.pi)

    if keypoints is None or len(keypoints) < 4:
        return 0.0

    angle1 = angle_between(keypoints[0], keypoints[1])
    angle2 = angle_between(keypoints[2], keypoints[3])
    cobb_angle = abs(angle1 - angle2)
    if cobb_angle > 180:
        cobb_angle = 360 - cobb_angle
    return float(cobb_angle)

def extract_landmarks_from_heatmaps_weighted(heatmaps, threshold=0.0, top_k=9):
    """
    Extract landmarks using weighted average of top-k highest values.
    
    This is a compromise between argmax (uses only 1 pixel) and full center-of-mass
    (uses all pixels). Often works well for heatmaps with localized peaks.
    
    Args:
        heatmaps: torch tensor or numpy array of shape (B, C, H, W), (C, H, W), or (H, W)
        threshold: Minimum peak value to consider (default: 0.0)
        top_k: Number of top pixels to average (default: 9, i.e., 3x3 region)
    
    Returns:
        list of length B, each element is a list of C (x,y) tuples or None when not found
    """
    import numpy as np
    import torch

    # Convert torch -> numpy
    if torch.is_tensor(heatmaps):
        heatmaps = heatmaps.detach().cpu().numpy()

    heatmaps = np.array(heatmaps)

    # Normalize to batch dimension
    if heatmaps.ndim == 4:
        batch_hm = heatmaps  # (B, C, H, W)
    elif heatmaps.ndim == 3:
        batch_hm = heatmaps[np.newaxis, ...]  # (1, C, H, W)
    elif heatmaps.ndim == 2:
        batch_hm = heatmaps[np.newaxis, np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported heatmaps shape: {heatmaps.shape}")

    all_landmarks = []
    for hm in batch_hm:  # hm: (C, H, W)
        c, h, w = hm.shape
        pts = []
        for i in range(c):
            ch = hm[i]
            
            # Check if peak exists
            if np.nanmax(ch) <= threshold:
                pts.append(None)
                continue
            
            # Get top-k pixel coordinates
            flat_indices = np.argpartition(ch.ravel(), -top_k)[-top_k:]
            top_values = ch.ravel()[flat_indices]
            
            # Convert flat indices to 2D coordinates
            y_coords, x_coords = np.unravel_index(flat_indices, ch.shape)
            
            # Weighted average using heatmap values as weights
            weights = top_values / (top_values.sum() + 1e-7)
            weighted_x = np.sum(x_coords * weights)
            weighted_y = np.sum(y_coords * weights)
            
            pts.append((float(weighted_x), float(weighted_y)))
                
        all_landmarks.append(pts)

    return all_landmarks


