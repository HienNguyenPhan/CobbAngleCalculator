import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch, hyperparams):
    """Train for one epoch"""
    model.train()
    criterion.train()
    
    running_loss = 0.0
    running_heatmap_loss = 0.0
    running_seg_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=hyperparams['use_amp']):
            predictions = model(images)
            losses = criterion(predictions, targets)
            loss = losses['total']
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), hyperparams['clip_grad_norm'])
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Accumulate losses
        running_loss += loss.item()
        running_heatmap_loss += losses['heatmap'].item()
        running_seg_loss += losses['segmentation'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'hm': f"{losses['heatmap'].item():.4f}",
            'sg': f"{losses['segmentation'].item():.4f}"
        })
    
    # Average losses
    epoch_loss = running_loss / len(dataloader)
    epoch_heatmap_loss = running_heatmap_loss / len(dataloader)
    epoch_seg_loss = running_seg_loss / len(dataloader)
    
    return {
        'loss': epoch_loss,
        'heatmap_loss': epoch_heatmap_loss,
        'seg_loss': epoch_seg_loss
    }


def validate(model, dataloader, criterion, device, epoch, hyperparams):
    """Validate the model"""
    from utils import get_coords_from_heatmap_gpu
    
    model.eval()
    criterion.eval()
    
    running_loss = 0.0
    running_heatmap_loss = 0.0
    running_seg_loss = 0.0
    running_dist_error = 0.0
    total_landmarks = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Val]')
    
    with torch.no_grad():
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.amp.autocast('cuda', enabled=hyperparams['use_amp']):
                predictions = model(images)
                losses = criterion(predictions, targets)
                loss = losses['total']
            
            # Accumulate losses
            running_loss += loss.item() * images.size(0)
            running_heatmap_loss += losses['heatmap'].item() * images.size(0)
            running_seg_loss += losses['segmentation'].item() * images.size(0)
            
            # Coordinate validation
            pred_hm = predictions[:, :4]
            true_hm = targets[:, :4]
            
            pred_coords = get_coords_from_heatmap_gpu(pred_hm)
            true_coords = get_coords_from_heatmap_gpu(true_hm)
            
            distances = torch.norm(pred_coords - true_coords, dim=2)
            valid_mask = (true_hm.view(true_hm.shape[0], true_hm.shape[1], -1).max(dim=2)[0] > 0.1)
            
            if valid_mask.sum() > 0:
                avg_batch_dist = distances[valid_mask].mean()
                running_dist_error += avg_batch_dist.item() * images.size(0)
                total_landmarks += images.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'hm': f"{losses['heatmap'].item():.4f}",
                'seg': f"{losses['segmentation'].item():.4f}",
            })
    
    n_samples = len(dataloader.dataset)
    epoch_loss = running_loss / n_samples
    epoch_heatmap_loss = running_heatmap_loss / n_samples
    epoch_seg_loss = running_seg_loss / n_samples
    epoch_dist_error = running_dist_error / (total_landmarks + 1e-7)
    
    return {
        'loss': epoch_loss,
        'distance_error': epoch_dist_error,
        'heatmap_loss': epoch_heatmap_loss,
        'seg_loss': epoch_seg_loss
    }


def save_checkpoint(model, optimizer, scheduler, criterion, training_state, epoch, filepath):
    """Save model checkpoint"""
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'training_state': {
            'best_val_loss': training_state.best_val_loss,
            'best_epoch': training_state.best_epoch,
            'train_losses': training_state.train_losses,
            'val_losses': training_state.val_losses,
            'learning_rates': training_state.learning_rates,
            'task_weights_history': training_state.task_weights_history
        },
    }
    torch.save(checkpoint, filepath)