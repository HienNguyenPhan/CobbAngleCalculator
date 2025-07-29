# Import
import torch
from data_loader import load_and_create_dataloader
from model import DeepLabV3Plus
from train import train_one_epoch
from eval import evaluate

DATA_DIR_JSON = "./data/default.json"
IMAGE_DIR = "./images/"

train_loader, val_loader, test_loader = load_and_create_dataloader(DATA_DIR_JSON, IMAGE_DIR)

model = DeepLabV3Plus()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

num_epochs = 200
patience = 8
epoch_without_improvement = 0
best_val_loss = float('inf')

for param in model.encoder.parameters():
    param.requires_grad = False

for epoch in range(num_epochs):
    if epoch == 10:
        print("Unfreezing encoder...")
        for param in model.encoder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, mean_nme = evaluate(model, val_loader, criterion)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | NME: {mean_nme:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epoch_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model!")
    else:
        epoch_without_improvement += 1
        if epoch_without_improvement > patience:
            break
        