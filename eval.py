import torch
import numpy as np
from utils import extract_keypoints_from_heatmap, get_keypoints_from_heatmap, compute_nme
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    nmes = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.cuda()
            targets = targets.cuda()
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            
            for i in range(images.size(0)):
                pred_heatmap = outputs[i, :6].cpu().numpy()
                gt_heatmap = targets[i, :4].cpu().numpy()

                pred_kps, _ = extract_keypoints_from_heatmap(pred_heatmap)
                gt_kps = get_keypoints_from_heatmap(gt_heatmap)

                nme = compute_nme(pred_kps, gt_kps)
                nmes.append(nme)
    return (running_loss / len(loader.dataset), np.mean(nmes))