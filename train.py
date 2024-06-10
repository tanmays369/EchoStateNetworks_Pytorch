import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import EchoStateNetwork
from dataset import MackeyGlassDataset

def train(model, train_loader, valid_loader, optimizer, device, epochs=100, early_stop=True, patience=5, path):
    best_loss= float('inf')
    epoch_no_improve= 0
    for epoch in range(epochs):
        model.train()
        train_loss= 0
        for x,y in train_loader:
            x, y= x.to(device), y.to(device)
            optimizer.zero_grad()
            output= model(x)
            loss= F.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)
        model.eval()
		
        val_loss= 0
        with torch.no_grad():
            for x,y in valid_loader:
                x,y= x.to(device), y.to(device)
                output= model(x)
                loss= F.mse_loss(output, y)
                val_loss += loss.item() * x.size(0)
        val_loss/= len(valid_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, \nTrain Loss: {train_loss:.6f}, \nValid Loss: {val_loss:.6f}')
        if early_stop:
            if val_loss < best_loss:
                best_loss= val_loss
                epoch_no_improve= 0
                torch.save(model.state_dict(), path)
            else:
                epoch_no_improve += 1
                if epoch_no_improve >= patience:
                    print(f'Early stopping after {patience} epochs')
                    break
