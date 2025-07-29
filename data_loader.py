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
        A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.05), rotate=(-10, 10), shear=(-5, 5), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485,), std=(0.229,)),
        ToTensorV2()
    ])

    transform_val = A.Compose([
        A.Normalize(mean=(0.485,), std=(0.229,  )),
        ToTensorV2()
    ])

    train_dataset = CervicalDataset(train_items, img_dir=IMG_PATH, transform=transform_train)
    val_dataset = CervicalDataset(val_items, img_dir=IMG_PATH, transform=transform_val)
    test_dataset = CervicalDataset(test_items, img_dir=IMG_PATH, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    

    return train_loader, val_loader, test_loader

