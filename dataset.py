import os
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class DogDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):

        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.breeds = sorted(self.df['breed'].unique())

        self.breed_to_idx = {b: i for i,b in enumerate(self.breeds)}
        self.idx_to_breed = {i: b for b,i in self.breed_to_idx.items()}

        self.df["label"] = self.df["breed"].map(self.breed_to_idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        img_id = self.df.iloc[idx]["id"]
        label = self.df.iloc[idx]["label"]

        path = os.path.join(self.img_dir, img_id + ".jpg")

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(train=True):

    if train:

        return transforms.Compose([

            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.RandomRotation(15),

            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )

        ])

    else:

        return transforms.Compose([

            transforms.Resize(256),
            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )

        ])


def get_dataloaders(csv_file,img_dir,batch_size=64,val_split=0.2):

    dataset = DogDataset(
        csv_file,
        img_dir,
        transform=get_transforms(True)
    )

    val_size = int(len(dataset)*val_split)
    train_size = len(dataset)-val_size

    train_dataset,val_dataset = random_split(dataset,[train_size,val_size])

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

    return train_loader,val_loader,dataset