import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from utils import (extract_landmarks_from_heatmaps_weighted, calculate_cobb_angle, 
                   calculate_cobb_angle_line_fitting)


def test_model(model, dataloader, criterion, device):
    """
    Advanced test function with line-fitting Cobb angle calculation.
    
    Args:
        model: Trained model
        dataloader: Test data loader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Dictionary with comprehensive test metrics
    """
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    running_heatmap_loss = 0.0
    running_seg_loss = 0.0
    landmark_errors = []
    angle_errors = []
    
    # For computing additional metrics
    all_heatmap_errors = []
    all_seg_dice_scores = []
    total_samples = 0
    
    print("=" * 60)
    print("TESTING MODEL - Line Fitting Cobb Angle")
    print("=" * 60)
    
    pbar = tqdm(dataloader, desc='Testing')
    eps = 1e-7
    
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            total_samples += images.size(0)
            
            # Forward pass
            predictions = model(images)
            losses = criterion(predictions, targets)
            
            # Accumulate losses
            running_loss += losses['total'].item()
            running_heatmap_loss += losses['heatmap'].item()
            running_seg_loss += losses['segmentation'].item()
            
            # ====================== Heatmaps & Landmarks ======================
            pred_heatmaps = predictions[:, :4]
            target_heatmaps = targets[:, :4]
            
            gt_landmarks = extract_landmarks_from_heatmaps_weighted(target_heatmaps)
            pred_landmarks = extract_landmarks_from_heatmaps_weighted(pred_heatmaps)
            
            # Compute landmark pixel errors
            for i in range(len(gt_landmarks)):
                gt_pts = gt_landmarks[i]
                pred_pts = pred_landmarks[i]
                for gt, pred in zip(gt_pts, pred_pts):
                    if gt is not None and pred is not None:
                        dist = np.linalg.norm(np.array(gt) - np.array(pred))
                        landmark_errors.append(float(dist))
            
            # ====================== Segmentation ======================
            pred_seg_logits = predictions[:, 4:11]
            pred_seg_class = torch.argmax(pred_seg_logits, dim=1)  # [B, H, W]
            
            # Correct target segmentation channel: only channel 4 stores class ids.
            target_seg = torch.nan_to_num(targets[:, 4], nan=0.0, posinf=0.0, neginf=0.0).round().clamp_(0, 6).long()
            
            # ====================== Cobb Angle (Line Fitting Method) ======================
            for i in range(pred_seg_class.shape[0]):
                pred_mask_np = pred_seg_class[i].detach().cpu().numpy()
                
                pred_kpts = pred_landmarks[i]
                gt_kpts = gt_landmarks[i]
                
                pred_angle = calculate_cobb_angle_line_fitting(pred_mask_np, pred_kpts)
                gt_angle = calculate_cobb_angle(gt_kpts)
                
                if pred_angle is not None and gt_angle is not None:
                    angle_errors.append(float(abs(pred_angle - gt_angle)))
            
            # ====================== Heatmap MSE ======================
            heatmap_error = F.mse_loss(pred_heatmaps, target_heatmaps, reduction='none').mean(dim=[1, 2, 3])
            all_heatmap_errors.extend(heatmap_error.detach().cpu().numpy().reshape(-1).tolist())
            
            # ====================== Foreground Dice ======================
            class_dices = []
            for cls in range(1, 7):
                pred_c = (pred_seg_class == cls)
                target_c = (target_seg == cls)
                
                intersection = (pred_c & target_c).sum(dim=[1, 2]).float()
                union = pred_c.sum(dim=[1, 2]).float() + target_c.sum(dim=[1, 2]).float()
                
                valid = union > 0
                if valid.any():
                    dice_c = (2.0 * intersection[valid] + eps) / (union[valid] + eps)
                    class_dices.append(dice_c)
            
            if class_dices:
                batch_dice = torch.cat(class_dices).mean()
                all_seg_dice_scores.extend(torch.cat(class_dices).detach().cpu().numpy().reshape(-1).tolist())
            else:
                batch_dice = torch.tensor(0.0, device=device)
            
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'dice_fg': f"{batch_dice.item():.4f}"
            })
    
    # ====================== Final Statistics ======================
    num_batches = len(dataloader)
    
    avg_loss = running_loss / num_batches
    avg_heatmap_loss = running_heatmap_loss / num_batches
    avg_seg_loss = running_seg_loss / num_batches
    
    avg_heatmap_error = np.mean(all_heatmap_errors) if all_heatmap_errors else 0.0
    avg_dice_score = np.mean(all_seg_dice_scores) if all_seg_dice_scores else 0.0
    
    # Comprehensive landmark error statistics
    if landmark_errors:
        landmark_errors_array = np.array(landmark_errors)
        mean_landmark_error = np.mean(landmark_errors_array)
        max_landmark_error = np.max(landmark_errors_array)
        min_landmark_error = np.min(landmark_errors_array)
        median_landmark_error = np.median(landmark_errors_array)
        std_landmark_error = np.std(landmark_errors_array)
        percentile_95_landmark = np.percentile(landmark_errors_array, 95)
    else:
        mean_landmark_error = max_landmark_error = min_landmark_error = median_landmark_error = std_landmark_error = percentile_95_landmark = 0.0
        landmark_errors_array = np.array([])
    
    # Comprehensive Cobb angle error statistics
    if angle_errors:
        angle_errors_array = np.array(angle_errors)
        mean_cobb_error = np.mean(angle_errors_array)
        max_cobb_error = np.max(angle_errors_array)
        min_cobb_error = np.min(angle_errors_array)
        median_cobb_error = np.median(angle_errors_array)
        std_cobb_error = np.std(angle_errors_array)
        percentile_95_angle = np.percentile(angle_errors_array, 95)
    else:
        mean_cobb_error = max_cobb_error = min_cobb_error = median_cobb_error = std_cobb_error = percentile_95_angle = 0.0
        angle_errors_array = np.array([])
    
    print("\nTEST RESULTS")
    print(f"Batches processed: {num_batches}")
    print(f"Samples processed: {total_samples}")
    print(f"Segmentation Loss: {avg_seg_loss:.6f}")
    print(f"\nHeatmap MSE:       {avg_heatmap_error:.6f}")
    print(f"Foreground Dice:   {avg_dice_score:.4f}")
    
    print(f"\nLANDMARK DISTANCE ERROR STATISTICS (pixels):")
    print(f"   Mean:       {mean_landmark_error:.2f}")
    print(f"   Median:     {median_landmark_error:.2f}")
    print(f"   Std Dev:    {std_landmark_error:.2f}")
    print(f"   Min:        {min_landmark_error:.2f}")
    print(f"   Max:        {max_landmark_error:.2f}")
    print(f"   95th %ile:  {percentile_95_landmark:.2f}")
    print(f"   Total evaluated: {len(landmark_errors)}")
    
    print(f"\nCOBB ANGLE ERROR STATISTICS (Line-fitting method):")
    print(f"   Mean:       {mean_cobb_error:.2f} deg")
    print(f"   Median:     {median_cobb_error:.2f} deg")
    print(f"   Std Dev:    {std_cobb_error:.2f} deg")
    print(f"   Min:        {min_cobb_error:.2f} deg")
    print(f"   Max:        {max_cobb_error:.2f} deg")
    print(f"   95th %ile:  {percentile_95_angle:.2f} deg")
    print(f"   Total evaluated: {len(angle_errors)}")
    print("=" * 60)
    
    return {
        'total_loss': avg_loss,
        'heatmap_loss': avg_heatmap_loss,
        'seg_loss': avg_seg_loss,
        'heatmap_error': avg_heatmap_error,
        'dice_score': avg_dice_score,
        'samples_processed': total_samples,
        'batches_processed': num_batches,
        'mean_landmark_error': mean_landmark_error,
        'mean_cobb_error': mean_cobb_error,
        'landmark_errors': landmark_errors_array if landmark_errors else np.array([]),
        'angle_errors': angle_errors_array if angle_errors else np.array([])
    }


def evaluate(model, loader, criterion):
    """Simplified evaluate function for compatibility"""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(next(model.parameters()).device)
            targets = targets.to(next(model.parameters()).device)
            outputs = model(images)
            if isinstance(outputs, dict):
                loss = outputs['total'] if 'total' in outputs else criterion(outputs['logits'], targets)
            else:
                loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)