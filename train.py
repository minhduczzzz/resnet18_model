import torch
import torch.nn as nn
import torch.optim as optim
import time

from torch.utils.data import DataLoader, random_split

from dataset import DogDataset, get_transforms
from model import ResNet34


csv_file = "labels.csv"
img_dir = "data/train"

batch_size = 64
epochs = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset
dataset = DogDataset(csv_file, img_dir, get_transforms(train=True))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataset.dataset.transform = get_transforms(True)
val_dataset.dataset.transform = get_transforms(False)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)


num_classes = len(dataset.breeds)

model = ResNet34(num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.003,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)


total_start = time.time()

for epoch in range(epochs):

    epoch_start = time.time()

    # ================= TRAIN =================
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        train_loss += loss.item() * images.size(0)

        _, pred = outputs.max(1)

        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc = 100 * correct / total


    # ================= VALID =================
    model.eval()

    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            _, pred = outputs.max(1)

            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * correct / total

    scheduler.step()

    epoch_time = time.time() - epoch_start

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.2f}% | "
        f"Time: {epoch_time:.1f}s"
    )


total_time = time.time() - total_start

print(f"\nTotal Training Time: {total_time/60:.2f} minutes")