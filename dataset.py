import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split


# ======================
# DATASET CLASS
# ======================
class DogDataset(Dataset):
    def __init__(self, df, img_dir, breed_to_idx, transform=None):
        """
        df            : pandas DataFrame (đã chứa id, breed)
        img_dir       : folder chứa ảnh
        breed_to_idx  : mapping breed -> label
        transform     : transform ảnh
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.breed_to_idx = breed_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_id = row['id']
        breed = row['breed']
        label = self.breed_to_idx[breed]

        img_path = os.path.join(self.img_dir, img_id + ".jpg")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ======================
# TRANSFORMS
# ======================

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize(256),

            # 🔥 crop ngẫu nhiên + scale
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),

            # 🔥 augment hình học
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),

            # 🔥 augment màu
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),

            # 🔥 thêm noise nhẹ
            transforms.RandomGrayscale(p=0.1),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),   # 🔥 chuẩn hơn Resize(224)

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ======================
# CREATE DATALOADER
# ======================
def get_dataloaders(csv_file, img_dir, batch_size=32, val_ratio=0.2):

    df = pd.read_csv(csv_file)

    # ===== LABEL MAPPING (CHUNG) =====
    breeds = sorted(df['breed'].unique())
    breed_to_idx = {breed: idx for idx, breed in enumerate(breeds)}
    idx_to_breed = {idx: breed for breed, idx in breed_to_idx.items()}

    # ===== SPLIT TRAIN / VAL =====
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        stratify=df['breed'],  # giữ cân bằng class
        random_state=42
    )

    # ===== DATASET =====
    train_dataset = DogDataset(
        train_df,
        img_dir,
        breed_to_idx,
        transform=get_transforms(train=True)
    )

    val_dataset = DogDataset(
        val_df,
        img_dir,
        breed_to_idx,
        transform=get_transforms(train=False)
    )

    # ===== DATALOADER =====
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, breed_to_idx, idx_to_breed


# ======================
# DEBUG FUNCTION (OPTIONAL)
# ======================
def debug_dataset(loader):
    images, labels = next(iter(loader))
    print("Batch shape:", images.shape)
    print("Labels:", labels[:10])