# Import
import torch
from data_loader import load_and_create_dataloader
from model import Model
from train import train_one_epoch
from eval import evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau

DATA_DIR_JSON = "./data/default.json"
IMAGE_DIR = "data/images/"

train_loader, val_loader, test_loader = load_and_create_dataloader(DATA_DIR_JSON, IMAGE_DIR)

model = Model()
model = model.cuda()

weights = torch.load("./imagenet.pth", map_location=torch.device('cuda')) 

if '_conv_stem.weight' in weights:
    conv_stem = weights['_conv_stem.weight']
    weights['_conv_stem.weight'] = conv_stem.mean(dim=1, keepdim=True)

weights = {k: v for k, v in weights.items() if not k.startswith('_fc')}

model_dict = model.encoder.state_dict()
filtered_weights = {}

for k, v in weights.items():
    if k in model_dict and model_dict[k].shape == v.shape:
        filtered_weights[k] = v

model.encoder.load_state_dict(filtered_weights, strict=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

num_epochs = 200
patience = 8
epoch_without_improvement = 0
best_nme = float('inf')    

for param in model.encoder.parameters():
    param.requires_grad = False

for epoch in range(num_epochs):
    if epoch == 10:
        print("Unfreezing encoder...")
        for param in model.encoder.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, nme = evaluate(model, val_loader, criterion)
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | NME: { nme:.4f}")

    if nme < best_nme:
        best_nme = nme
        epoch_without_improvement = 0
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved best model!")
    else:
        epoch_without_improvement += 1
        if epoch_without_improvement > patience:
            break
        