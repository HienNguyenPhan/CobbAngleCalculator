import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from model import CervicalMultiTaskTransformer
from dataset import CervicalDataset
from train import train_one_epoch, validate, save_checkpoint
from eval import test_model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalMSELoss(nn.Module):
    """Focal MSE for heatmap regression - emphasizes harder samples"""
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        mse = (pred - target) ** 2
        focal_weight = (1 - torch.exp(-self.beta * mse)) ** self.alpha
        loss = (focal_weight * mse).mean()
        return loss


class ImprovedMultiTaskLoss(nn.Module):
    """
    Multi-Task Loss for 4 heatmaps + 7-class segmentation
    - Heatmaps: Focal MSE
    - Segmentation: CrossEntropyLoss + optional Dice
    """
    def __init__(self, use_adaptive_wing=False, dice_weight=0.5):
        super().__init__()
        
        # Learnable uncertainty weighting (2 tasks)
        initial_log_vars = torch.tensor([0.0, 1.0])   # heatmap, segmentation
        self.log_vars = nn.Parameter(initial_log_vars)
        
        # Heatmap loss
        self.heatmap_loss_fn = FocalMSELoss(alpha=2.0, beta=4.0)
        print("Using Focal MSE Loss for heatmaps")
        
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        """
        predictions: [B, 11, H, W]  -> channels 0-3: heatmaps, channels 4-10: seg logits for 7 classes
        targets:     [B, 5, H, W]   -> channels 0-3: heatmaps, channel 4: class ids (0..6)
        """
        # Split
        pred_heatmaps = predictions[:, :4]              # [B, 4, H, W]
        pred_seg_logits = predictions[:, 4:11]          # [B, 7, H, W]
        
        target_heatmaps = targets[:, :4]                # [B, 4, H, W]
        target_seg = targets[:, 4].long()               # [B, H, W], CE requires int64 class indices

        # 1. Heatmap Loss
        loss_heatmap = self.heatmap_loss_fn(pred_heatmaps, target_heatmaps)

        # 2. Segmentation Loss - Multi-class
        loss_ce = F.cross_entropy(
            pred_seg_logits, 
            target_seg, 
            ignore_index=0,
            reduction='mean'
        )

        # Optional Multi-class Dice
        loss_dice = torch.tensor(0.0, device=predictions.device)
        if self.dice_weight > 0:
            pred_seg_prob = F.softmax(pred_seg_logits, dim=1)   # [B, 7, H, W]
            target_onehot = F.one_hot(target_seg, num_classes=7).permute(0, 3, 1, 2).float()
            
            intersection = (pred_seg_prob * target_onehot).sum(dim=(2, 3))
            union = pred_seg_prob.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3))
            dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
            loss_dice = 1 - dice.mean()

        loss_seg = loss_ce + self.dice_weight * loss_dice

        # Uncertainty weighting (homoscedastic uncertainty)
        precision_heatmap = torch.exp(-self.log_vars[0])
        precision_seg = torch.exp(-self.log_vars[1])

        weighted_heatmap = precision_heatmap * loss_heatmap + self.log_vars[0]
        weighted_seg = precision_seg * loss_seg + self.log_vars[1]

        total_loss = weighted_heatmap + weighted_seg

        losses = {
            'total': total_loss,
            'heatmap': loss_heatmap.detach(),
            'segmentation': loss_seg.detach(),
            'seg_ce': loss_ce.detach(),
            'seg_dice': loss_dice.detach(),
            'weight_heatmap': precision_heatmap.detach(),
            'weight_seg': precision_seg.detach(),
        }

        return losses


class ImprovedTrainingState:
    """Track training progress and state"""
    def __init__(self):
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.task_weights_history = {'heatmap': [], 'seg': []}
        self.heatmap_losses = []
        self.seg_losses = []
        
    def update(self, epoch, train_loss, val_loss, lr, task_weights, heatmap_loss=None, seg_loss=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        self.task_weights_history['heatmap'].append(task_weights[0])
        self.task_weights_history['seg'].append(task_weights[1])
        
        if heatmap_loss is not None:
            self.heatmap_losses.append(heatmap_loss)
        if seg_loss is not None:
            self.seg_losses.append(seg_loss)
        
        # Check for improvement
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            return True  # Improved
        else:
            self.epochs_no_improve += 1
            return False  # No improvement
    
    def should_stop(self, patience):
        return self.epochs_no_improve >= patience
    
    def print_summary(self):
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Best Val Loss: {self.best_val_loss:.6f} at epoch {self.best_epoch + 1}")
        print(f"Total epochs trained: {len(self.train_losses)}")
        if len(self.heatmap_losses) > 0:
            print(f"Final Heatmap Loss: {self.heatmap_losses[-1]:.6f}")
        if len(self.seg_losses) > 0:
            print(f"Final Seg Loss: {self.seg_losses[-1]:.6f}")
        print(f"Final Task Weights:")
        print(f"  Heatmap: {self.task_weights_history['heatmap'][-1]:.4f}")
        print(f"  Seg: {self.task_weights_history['seg'][-1]:.4f}")
        print("="*80)


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

HYPERPARAMS = {
    # Training
    'epochs': 150,
    'batch_size': 8,
    'image_size': (256, 256),
    
    # Optimizer
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
    
    # Scheduler
    'scheduler_type': 'cosine',  # 'cosine' or 'plateau'
    'min_lr': 1e-6,
    'patience': 10,  # for ReduceLROnPlateau
    'factor': 0.5,   # for ReduceLROnPlateau
    
    # Early stopping
    'early_stop_patience': 20,
    
    # Model
    'encoder_name': 'mit_b2',
    'encoder_weights': None,
    
    # Loss
    'use_adaptive_wing': False,
    
    # Regularization
    'dropout': 0.1,
    'label_smoothing': 0.0,
    
    # Gradient clipping
    'clip_grad_norm': 1.0,
    
    # Mixed precision training
    'use_amp': True,
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'pin_memory': True,
    
    # Checkpointing
    'save_dir': './checkpoints',
    'save_best_only': True,
    'monitor_metric': 'val_loss',
}


def plot_training_history(training_state, save_dir='./checkpoints'):
    """Visualize training progress"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(training_state.train_losses) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, training_state.train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, training_state.val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].axvline(x=training_state.best_epoch + 1, color='g', linestyle='--', 
                       label=f'Best (Epoch {training_state.best_epoch + 1})')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[0, 1].plot(epochs, training_state.learning_rates, 'purple', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[0, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Task weights evolution
    axes[1, 0].plot(epochs, training_state.task_weights_history['heatmap'], 
                    label='Heatmap', linewidth=2)
    axes[1, 0].plot(epochs, training_state.task_weights_history['seg'], 
                    label='Segmentation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Task Weight (σ²)', fontsize=12)
    axes[1, 0].set_title('Learned Task Weights', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss improvement
    best_so_far = []
    current_best = float('inf')
    for val_loss in training_state.val_losses:
        if val_loss < current_best:
            current_best = val_loss
        best_so_far.append(current_best)
    
    axes[1, 1].plot(epochs, training_state.val_losses, 'r-', alpha=0.5, label='Val Loss')
    axes[1, 1].plot(epochs, best_so_far, 'g-', linewidth=2, label='Best Val Loss')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Validation Loss', fontsize=12)
    axes[1, 1].set_title('Validation Loss Improvement', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Training history plot saved to {save_path}")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Setup
    device = torch.device(HYPERPARAMS['device'])
    os.makedirs(HYPERPARAMS['save_dir'], exist_ok=True)
    
    print("="*80)
    print("CERVICAL SPINE LANDMARK DETECTION & SEGMENTATION")
    print("="*80)
    print("\nHYPERPARAMETERS:")
    for key, value in HYPERPARAMS.items():
        print(f"  {key:25s}: {value}")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    with open('./data/default_corner_masks.json') as f:
        data = json.load(f)
    items = data['items']
    
    train_val_items, test_items = train_test_split(items, test_size=0.1, random_state=42)
    train_items, val_items = train_test_split(train_val_items, test_size=0.1, random_state=42)
    
    # Define augmentations
    transform_train = A.Compose([
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), 
                rotate=(-15, 15), shear=(-5, 5), p=0.8),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(std_range=(0.1, 0.5), mean_range=(0, 0), per_channel=False, p=0.2),
        A.OneOf([A.GaussianBlur(blur_limit=3, p=0.5), A.MotionBlur(blur_limit=3, p=0.5)], p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2()
    ])
    
    transform_val = A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2()
    ])
    
    # Create datasets and dataloaders
    train_dataset = CervicalDataset(train_items, img_dir='./data/images/images', 
                                    image_size=HYPERPARAMS['image_size'], 
                                    transform=transform_train, include_masks=True)
    val_dataset = CervicalDataset(val_items, img_dir='./data/images/images', 
                                  image_size=HYPERPARAMS['image_size'], 
                                  transform=transform_val, include_masks=True)
    test_dataset = CervicalDataset(test_items, img_dir='./data/images/images', 
                                   image_size=HYPERPARAMS['image_size'], 
                                   transform=transform_val, include_masks=True)
    
    train_loader = DataLoader(train_dataset, batch_size=HYPERPARAMS['batch_size'], 
                             shuffle=True, num_workers=HYPERPARAMS['num_workers'], 
                             pin_memory=HYPERPARAMS['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=HYPERPARAMS['batch_size'], 
                           shuffle=False, num_workers=HYPERPARAMS['num_workers'], 
                           pin_memory=HYPERPARAMS['pin_memory'])
    test_loader = DataLoader(test_dataset, batch_size=HYPERPARAMS['batch_size'], 
                            shuffle=False, num_workers=HYPERPARAMS['num_workers'], 
                            pin_memory=HYPERPARAMS['pin_memory'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    print("\nInitializing model...")
    model = CervicalMultiTaskTransformer(
        encoder_name=HYPERPARAMS['encoder_name'],
        encoder_weights=HYPERPARAMS['encoder_weights']
    )
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = ImprovedMultiTaskLoss(use_adaptive_wing=False, dice_weight=0.5)
    criterion = criterion.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS['learning_rate'],
        weight_decay=HYPERPARAMS['weight_decay'],
        betas=HYPERPARAMS['betas']
    )
    
    if HYPERPARAMS['scheduler_type'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=HYPERPARAMS['epochs'],
            eta_min=HYPERPARAMS['min_lr']
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=HYPERPARAMS['factor'],
            patience=HYPERPARAMS['patience'],
            min_lr=HYPERPARAMS['min_lr'],
            verbose=True
        )
    
    scaler = torch.amp.GradScaler('cuda', enabled=HYPERPARAMS['use_amp'])
    training_state = ImprovedTrainingState()
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    for epoch in range(HYPERPARAMS['epochs']):
        # Unfreeze encoder at epoch 15
        if epoch == 15:
            print("\n🔓 UNFREEZING ENCODER with reduced learning rate")
            for param in model.encoder.parameters():
                param.requires_grad = True
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, HYPERPARAMS
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader, criterion, device, epoch, HYPERPARAMS
        )
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Get current task weights
        with torch.no_grad():
            task_weights = [
                criterion.log_vars[0].exp().item(),
                criterion.log_vars[1].exp().item()
            ]
        
        # Update training state
        improved = training_state.update(
            epoch, 
            train_metrics['loss'], 
            val_metrics['distance_error'],
            current_lr,
            task_weights
        )
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Train Loss: {train_metrics['loss']:.6f} | Val Loss: {val_metrics['loss']:.6f}")
        print(f"  - Heatmap  : Train {train_metrics['heatmap_loss']:.6f} | Val {val_metrics['heatmap_loss']:.6f}")
        print(f"  - Segmentation: Train {train_metrics['seg_loss']:.6f} | Val {val_metrics['seg_loss']:.6f}")
        print(f"Task Weights: Heatmap={task_weights[0]:.4f}, Seg={task_weights[1]:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")
        
        # Save checkpoint
        if improved:
            checkpoint_path = os.path.join(HYPERPARAMS['save_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, criterion, training_state, epoch, checkpoint_path)
            print(f"✓ Improved! New Best Distance Error: {val_metrics['distance_error']:.2f} pixels")
        
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(HYPERPARAMS['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, criterion, training_state, epoch, checkpoint_path)
            print(f"✓ Checkpoint saved")
        
        # Learning rate scheduling
        if HYPERPARAMS['scheduler_type'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_metrics['loss'])
        
        # Early stopping
        if training_state.should_stop(HYPERPARAMS['early_stop_patience']):
            print(f"\nEARLY STOPPING at epoch {epoch + 1}")
            break
    
    # Training completed
    training_state.print_summary()
    plot_training_history(training_state, HYPERPARAMS['save_dir'])
    
    # Test
    print("\n" + "="*80)
    print("TESTING ON TEST SET")
    print("="*80)
    test_results = test_model(model, test_loader, criterion, device)