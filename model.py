import torch
import torch.nn as nn


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,3,stride,1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,3,1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride!=1 or in_channels!=out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,stride,bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self,x):

        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(identity)

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,block,layers,num_classes):

        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(3,64,7,2,3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(3,2,1)

        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],2)
        self.layer3 = self._make_layer(block,256,layers[2],2)
        self.layer4 = self._make_layer(block,512,layers[3],2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(512,num_classes)

    def _make_layer(self,block,out_channels,blocks,stride=1):

        layers = []

        layers.append(block(self.in_channels,out_channels,stride))
        self.in_channels = out_channels

        for _ in range(1,blocks):
            layers.append(block(self.in_channels,out_channels))

        return nn.Sequential(*layers)

    def forward(self,x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = torch.flatten(x,1)

        x = self.dropout(x)

        x = self.fc(x)

        return x


def ResNet34(num_classes):
    return ResNet(BasicBlock,[3,4,6,3],num_classes)