import hashlib
import random

import numpy as np
import torch
from torchvision import models

from resnet import ResNet18, ResNet50


class TinyImageNetResNet50(models.resnet.ResNet):
    def __init__(self, classes):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        features = self.avgpool(x).flatten(1)
        return self.fc(features), features


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def make_model(name, classes, dataset=None):
    if dataset == "tiny-imagenet" and name == "r50":
        return TinyImageNetResNet50(classes)
    return {"r18": ResNet18, "r50": ResNet50}[name](
        num_classes=classes, return_features=True
    )


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()
