import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class TaskInteractionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 4, 1)
        self.key = nn.Conv2d(channels, channels // 4, 1)
        self.value = nn.Conv2d(channels, channels, 1)

        self.gamma = nn.Parameter(torch.full((1,), 0.01))
        
    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        
        query = self.query(x1).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x2).view(B, -1, H * W)
        
        # Scaling the dot product (similar to standard Transformer)
        scaling = (C // 4) ** -0.5
        attention = torch.softmax(torch.bmm(query, key) * scaling, dim=-1)
        
        value = self.value(x2).view(B, C, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return self.gamma * out + x1

class TaskSpecificHead(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, task_type='heatmap'):
        super().__init__()
        self.refine_skip = nn.Conv2d(skip_channels, 64, 1)
        
        # Final refinement layers
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels + 64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )
        
    def forward(self, x, skip):
        # Align skip features with current feature map size
        skip = self.refine_skip(skip)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        combined = torch.cat([x, skip], dim=1)
        return self.decoder(combined)

class CervicalMultiTaskTransformer(nn.Module):
    def __init__(self, encoder_name='mit_b2', encoder_weights='imagenet'):
        super().__init__()
        
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=1, depth=5, weights=encoder_weights
        )
        
        # MiT-B2 channels: [0, 64, 128, 320, 512] (typically)
        ch = self.encoder.out_channels
        
        # Multi-Scale Fusion: Combine all encoder features to 256 channels
        # We target the 1/4 resolution (the first encoder feature map size)
        self.fusion_conv = nn.Conv2d(sum(ch[2:]), 256, 1)
        
        # Shared bottleneck processing
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Task Interaction
        self.heatmap_interaction = TaskInteractionModule(128)
        self.seg_interaction = TaskInteractionModule(128)

        # Heads (Heatmap output=4, Seg output=1)
        # Using ch[1] as the skip connection (usually 1/4 resolution)
        self.heatmap_head = TaskSpecificHead(128, ch[2], 4)
        self.seg_head = TaskSpecificHead(128, ch[2], 1)
        
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        input_size = x.shape[-2:]
        features = self.encoder(x) # [f0=input, f1=1/4, f2=1/8, f3=1/16, f4=1/32]
        
        # Hierarchical Fusion (Point 1)
        # Upsample all features to the size of features[1] (1/4 scale)
        target_size = features[2].shape[-2:]
        fused_list = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) 
                      for f in features[2:]]
        
        fused = torch.cat(fused_list, dim=1)
        fused = self.fusion_conv(fused)
        
        # Shared processing
        shared = self.bottleneck(self.dropout(fused))
        
        # Interaction (Point 3)
        h_feat = self.heatmap_interaction(shared, shared)
        s_feat = self.seg_interaction(shared, shared)
        
        # Heads with Skip Connections (Point 2)
        heatmaps = self.heatmap_head(h_feat, features[2])
        seg_mask = self.seg_head(s_feat, features[2])
        
        # Final scale-up to original input resolution
        heatmaps = F.interpolate(heatmaps, size=input_size, mode='bilinear', align_corners=False)
        seg_mask = F.interpolate(seg_mask, size=input_size, mode='bilinear', align_corners=False)
        
        return torch.cat([heatmaps, seg_mask], dim=1)