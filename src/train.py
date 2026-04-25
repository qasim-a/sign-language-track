import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

X = np.load("data/X.npy")
y = np.load("data/y.npy")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# 80/20 train/validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16)

class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        # LSTM processes the sequence of landmark frames over time
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, len(SIGNS))

    def forward(self, x):
        out, _ = self.lstm(x)
        # take the output of the last frame only
        out = self.fc(out[:, -1, :])
        return out

SIGNS = ["hello", "yes", "no", "nothing"]
model = ASLModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            output = model(xb)
            preds = torch.argmax(output, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}")

torch.save(model.state_dict(), "models/asl_model.pth")
print("Model saved.")