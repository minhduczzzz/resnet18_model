import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DogDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        # tạo label index
        self.breeds = sorted(self.df['breed'].unique())
        self.breed_to_idx = {breed: i for i, breed in enumerate(self.breeds)}

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
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485,0.456,0.406],
                [0.229,0.224,0.225]
            )
        ])

    return transform