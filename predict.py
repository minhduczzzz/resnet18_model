import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

from model import ResNet18


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# load model
model = ResNet18(num_classes=120)

model.load_state_dict(torch.load("model.pth", map_location=device))

model = model.to(device)

model.eval()


# transform giống lúc test
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0).to(device)


    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

    return predicted.item()



if __name__ == "__main__":

    img_path = "test.jpg"

    pred = predict(img_path)

    print("Predicted class:", pred)