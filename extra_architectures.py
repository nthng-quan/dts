import torch
from torch import nn
from torchvision.models import convnext_tiny, efficientnet_b0
from torchvision.models.vision_transformer import VisionTransformer


class ViTSmall(nn.Module):
    def __init__(self, num_classes, image_size):
        super().__init__()
        patch_size = 4 if image_size == 32 else 8
        self.net = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=6,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=1536,
            num_classes=num_classes,
            dropout=0.0,
            attention_dropout=0.0,
        )

    def forward(self, x):
        x = self.net._process_input(x)
        cls = self.net.class_token.expand(x.shape[0], -1, -1)
        x = self.net.encoder(torch.cat([cls, x], dim=1))
        z = x[:, 0]
        return self.net.heads(z), z


class ConvNeXtTiny(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = convnext_tiny(weights=None, num_classes=num_classes)

    def forward(self, x):
        x = self.net.features(x)
        pooled = self.net.avgpool(x)
        z = pooled.flatten(1)
        return self.net.classifier(pooled), z


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = efficientnet_b0(weights=None, num_classes=num_classes)

    def forward(self, x):
        x = self.net.features(x)
        pooled = self.net.avgpool(x)
        z = pooled.flatten(1)
        return self.net.classifier(z), z


def make_extra_model(name, num_classes, image_size):
    if name == "vit_s":
        return ViTSmall(num_classes, image_size)
    if name == "convnext_tiny":
        return ConvNeXtTiny(num_classes)
    if name == "efficientnet_b0":
        return EfficientNetB0(num_classes)
    raise ValueError(f"Unknown extra architecture: {name}")
