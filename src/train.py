import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ── Reproducibility ───────────────────────────────────────────────────────────
# Fixed seed ensures identical train/val split and weight initialization
# every retrain on the same data. Change SEED to try a different split.
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ────────────────────────────────────────────────────────────────────
SIGNS = ["hello", "yes", "no", "nothing", "thank you", "please", "drink", "water", "more", "apple", "mother", "father", "book", "walk", "cold", "go", "break", "cow", "monkey", "draw"]
EPOCHS = 100
LR     = 0.0003
BATCH  = 16

# ── Data ──────────────────────────────────────────────────────────────────────
X = np.load("data/X.npy")
y = np.load("data/y.npy")

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor  = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)

# 80/20 train/validation split — deterministic with fixed seed
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))
val_loader   = DataLoader(val_set, batch_size=BATCH)

# ── Model ─────────────────────────────────────────────────────────────────────
class ASLModel(nn.Module):
    def __init__(self):
        super(ASLModel, self).__init__()
        # LSTM processes the sequence of landmark frames over time
        self.lstm = nn.LSTM(input_size=225, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.3)
        self.fc = nn.Linear(128, len(SIGNS))

    def forward(self, x):
        out, _ = self.lstm(x)
        # take the output of the last frame only
        out = self.fc(out[:, -1, :])
        return out

model     = ASLModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ── Training loop ─────────────────────────────────────────────────────────────
# saves the best model whenever val acc improves — protects against late spikes
best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        output = model(xb)
        loss   = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # validation pass every epoch
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            output = model(xb)
            preds  = torch.argmax(output, dim=1)
            correct += (preds == yb).sum().item()
            total   += yb.size(0)

    val_acc = correct / total

    # save whenever val acc improves — best model, not last model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "models/asl_model.pth")
        saved_marker = " Saved"
    else:
        saved_marker = ""

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.2f}{saved_marker}")

print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

# ── Confusion matrix (full validation pass on best saved model) ───────────────
# reload best weights before evaluating — ensures matrix reflects best model
model.load_state_dict(torch.load("models/asl_model.pth"))
model.eval()
all_preds  = []
all_labels = []

with torch.no_grad():
    for xb, yb in val_loader:
        output = model(xb)
        preds  = torch.argmax(output, dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(yb.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=SIGNS, yticklabels=SIGNS,
            cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (best model, seed {SEED}) — best val acc: {best_val_acc:.4f}")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()
print("Confusion matrix saved to models/confusion_matrix.png")
print(f"Model saved. (seed={SEED}, lr={LR}, epochs={EPOCHS}, best_val_acc={best_val_acc:.4f})")