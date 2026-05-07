import numpy as np
import torch

def angle_between(p1, p2):
    """Calculate angle between two points."""
    delta = np.array(p2) - np.array(p1)
    angle = np.arctan2(delta[1], delta[0]) * 180 / np.pi
    return angle

def extract_lower_edge_points(mask, ref_points, thickness=12, min_points=10):
    """
    More robust lower edge point extraction.
    Uses the two reference keypoints and searches in a band around the line.
    
    Args:
        mask: 2D segmentation mask for a single vertebra
        ref_points: Two reference keypoints (anterior & posterior endpoints)
        thickness: Band thickness for searching
        min_points: Minimum points required
    
    Returns:
        Nx2 array of edge points, or empty array if insufficient
    """
    if len(ref_points) < 2 or ref_points[0] is None or ref_points[1] is None:
        return np.array([])
    
    p1 = np.array(ref_points[0])
    p2 = np.array(ref_points[1])
    
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 5:
        return np.array([])
    
    unit_dir = direction / length
    # Perpendicular vector (both directions)
    perp = np.array([-unit_dir[1], unit_dir[0]])
    
    points = []
    
    # Sample densely along the reference line
    for t in np.linspace(0, 1, num=60):
        base = p1 + t * direction
        
        # Search in a thicker band on BOTH sides of the line
        for side in [-1, 1]:
            for d in np.linspace(0, thickness, num=thickness+1):
                test_pt = base + side * d * perp
                x = int(round(test_pt[0]))
                y = int(round(test_pt[1]))
                
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        points.append([x, y])
    
    points = np.array(points)
    
    # Remove duplicates if any
    if len(points) > 0:
        points = np.unique(points, axis=0)
    
    if len(points) < min_points:
        return np.array([])
    
    return points


def fit_line_least_squares(points):
    """
    Fit a line y = mx + c using least squares.
    
    Args:
        points: Nx2 array of (x, y) coordinates
    
    Returns:
        Tuple of (slope m, intercept c), or (None, None) if fit fails
    """
    if len(points) < 2:
        return None, None
    
    x = points[:, 0]
    y = points[:, 1]
    
    # Robust fit: use np.polyfit (degree 1)
    try:
        m, c = np.polyfit(x, y, 1)
        return m, c
    except:
        return None, None


def angle_from_slope(m):
    """
    Convert slope to angle in degrees [0, 180).
    
    Args:
        m: Slope value
    
    Returns:
        Angle in degrees, or None if m is None
    """
    if m is None:
        return None
    angle = np.degrees(np.arctan(m))
    angle = angle % 180
    return float(angle)


def angle_diff_undirected(a, b):
    """
    Compute smallest undirected angle difference between two lines.
    
    Args:
        a, b: Angles in degrees
    
    Returns:
        Smallest angle difference in [0, 90] degrees
    """
    if a is None or b is None:
        return None
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def calculate_cobb_angle_line_fitting(seg_mask, keypoints, c2_class=1, c7_class=6):
    """
    Advanced Cobb angle calculation using line fitting on lower vertebral edges.
    Uses segmentation mask and keypoints as reference for robust angle estimation.
    
    Args:
        seg_mask: 2D segmentation mask with class ids (0=bg, 1=C2, ..., 6=C7)
        keypoints: List of 4 keypoints [C2_left, C2_right, C7_left, C7_right]
        c2_class: Class id for C2 vertebra
        c7_class: Class id for C7 vertebra
    
    Returns:
        Cobb angle in degrees, or None if calculation fails
    """
    try:
        # Split keypoints
        c2_lower_corners = keypoints[:2]   # anterior & posterior of C2 lower edge
        c7_lower_corners = keypoints[2:]   # anterior & posterior of C7 lower edge
        
        # Get binary masks
        c2_mask = (seg_mask == c2_class).astype(np.uint8)
        c7_mask = (seg_mask == c7_class).astype(np.uint8)
        
        # Extract point sets along lower edges using keypoints as reference
        pc2 = extract_lower_edge_points(c2_mask, c2_lower_corners, thickness=3, min_points=15)
        pc7 = extract_lower_edge_points(c7_mask, c7_lower_corners, thickness=3, min_points=15)
        
        if len(pc2) < 15 or len(pc7) < 15:
            return None
        
        # Fit lines
        m2, _ = fit_line_least_squares(pc2)
        m7, _ = fit_line_least_squares(pc7)
        
        angle_c2 = angle_from_slope(m2)
        angle_c7 = angle_from_slope(m7)
        
        return angle_diff_undirected(angle_c2, angle_c7)
    except Exception as e:
        return None


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

