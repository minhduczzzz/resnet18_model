import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DogDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        csv_file : path to labels.csv
        img_dir  : folder chứa ảnh train
        transform: image transform
        """
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # tạo danh sách breed
        self.breeds = sorted(self.df['breed'].unique())

        # map breed -> index
        self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
        self.idx_to_breed = {idx: breed for breed, idx in self.breed_to_idx.items()}

        # thêm column label dạng số
        self.df['label'] = self.df['breed'].map(self.breed_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_id = self.df.iloc[idx]['id']
        label = self.df.iloc[idx]['label']

        img_path = os.path.join(self.img_dir, img_id + ".jpg")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train=True):

    if train:
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    return transform


def get_dataloader(csv_file, img_dir, batch_size=32, train=True):

    dataset = DogDataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=get_transforms(train)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2
    )

    return loader, dataset