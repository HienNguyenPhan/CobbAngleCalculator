from sklearn.model_selection import train_test_split
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import CervicalDataset
from torch.utils.data import DataLoader

def load_and_create_dataloader(DATA_DIR_JSON, IMG_PATH):
    with open(DATA_DIR_JSON) as f:
        data = json.load(f)
    items = data['items']

    # Splitting train, val, test at 80/10/10
    train_val_items, test_items = train_test_split(items, test_size=0.1, random_state=42)
    train_items, val_items = train_test_split(train_val_items, test_size=0.1, random_state=42)

    # Transform
    transform_train = A.Compose([
        A.Affine(
            scale=(0.9, 1.1), 
            translate_percent=(0.05, 0.05), 
            rotate=(-15, 15),
            shear=(-5, 5), 
            p=0.8
        ),
        A.HorizontalFlip(p=0.5),
        
        A.RandomBrightnessContrast(
            brightness_limit=0.2, 
            contrast_limit=0.2, 
            p=0.4
        ),

        A.CLAHE(
            clip_limit=2.0, 
            tile_grid_size=(8, 8), 
            p=0.3
        ),
        
        A.GaussNoise(
            std_range=(0.1, 0.5),
            mean_range=(0, 0),
            per_channel=False,
            p=0.2
        ),
        
        A.OneOf([
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        
        A.RandomGamma(
            gamma_limit=(80, 120), 
            p=0.2
        ),
        
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2()
])


    transform_val = A.Compose([
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # Always apply CLAHE
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2()
    ])


    train_dataset = CervicalDataset(train_items, img_dir=IMG_PATH, transform=transform_train)
    val_dataset = CervicalDataset(val_items, img_dir=IMG_PATH, transform=transform_val)
    test_dataset = CervicalDataset(test_items, img_dir=IMG_PATH, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=2, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2, shuffle=False, pin_memory=True)

    return train_loader, val_loader, test_loader

