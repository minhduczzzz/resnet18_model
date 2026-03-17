import time
import torch
import torch.nn as nn
import torch.optim as optim

from model import ResNet18
from dataset import get_dataloaders   # 🔥 IMPORT Ở ĐÂY

# ======================
# CONFIG
# ======================
csv_file = "labels.csv"
img_dir = "data/train"
batch_size = 32
num_epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# LOAD DATA 🔥
# ======================
trainloader, testloader, breed_to_idx, idx_to_breed = get_dataloaders(
    csv_file=csv_file,
    img_dir=img_dir,
    batch_size=batch_size
)

print("Num classes:", len(breed_to_idx))

# ======================
# MODEL
# ======================
model = ResNet18(num_classes=len(breed_to_idx)).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ======================
# TRAIN
# ======================
for epoch in range(num_epochs):
    start = time.time()

    # ===== TRAIN =====
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for inputs, labels in trainloader:   # 🔥 dùng từ dataset.py
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(trainloader.dataset)
    train_acc = 100. * correct / total

    # ===== VALIDATION =====
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for inputs, labels in testloader:   # 🔥 dùng từ dataset.py
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(testloader.dataset)
    val_acc = 100. * correct / total

    scheduler.step()

    end = time.time()

    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Val Acc: {val_acc:.2f}% | "
          f"Time: {end-start:.2f}s")

# ======================
# SAVE MODEL
# ======================
torch.save(model.state_dict(), "dog_model.pth")