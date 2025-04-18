
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import DualTransferLungClassifier
from tqdm import tqdm
import os

def train_model(data_dir, epochs=50, batch_size=32, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _ = get_dataloaders(data_dir, batch_size=batch_size)
    
    model = DualTransferLungClassifier(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    os.makedirs('saved_models', exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, leave=True)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        val_loss = evaluate_model(model, val_loader, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print(f"Saved Best Model at Epoch {epoch+1}! Validation Loss: {val_loss:.4f}")

def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to LC25000 dataset')
    args = parser.parse_args()
    train_model(data_dir=args.data_path)
