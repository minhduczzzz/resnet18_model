import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from dataset import get_dataloaders
from model import ResNet34


csv_file = "labels.csv"
img_dir = "data/train"

epochs = 100
batch_size = 64

train_loader,val_loader,dataset = get_dataloaders(
    csv_file,
    img_dir,
    batch_size
)

num_classes = len(dataset.breeds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet34(num_classes).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs
)


def mixup_data(x,y,alpha=0.4):

    lam = np.random.beta(alpha,alpha)

    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam*x + (1-lam)*x[index]

    y_a,y_b = y,y[index]

    return mixed_x,y_a,y_b,lam


for epoch in range(epochs):

    model.train()

    correct = 0
    total = 0

    for images,labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        images,y_a,y_b,lam = mixup_data(images,labels)

        optimizer.zero_grad()

        outputs = model(images)

        loss = lam*criterion(outputs,y_a) + (1-lam)*criterion(outputs,y_b)

        loss.backward()

        optimizer.step()

        _,pred = outputs.max(1)

        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    train_acc = 100*correct/total

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images,labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _,pred = outputs.max(1)

            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    val_acc = 100*correct/total

    scheduler.step()

    print(
        f"Epoch {epoch+1}/{epochs} "
        f"Train Acc {train_acc:.2f}% "
        f"Val Acc {val_acc:.2f}%"
    )

torch.save(model.state_dict(),"dog_resnet34.pth")