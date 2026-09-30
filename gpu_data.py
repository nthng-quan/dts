import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms

from data import SPECS, TinyImageNet


class CIFARGPUBatches:
    def __init__(self, name, root, device, seed):
        self.name = name
        if name in ("cifar10", "cifar100"):
            cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
            dataset = cls(str(root), train=True, download=False)
            images = torch.from_numpy(np.asarray(dataset.data))
            targets = dataset.targets
        elif name == "tiny-imagenet":
            dataset = TinyImageNet(root, train=True, transform=transforms.PILToTensor())
            images = torch.stack([dataset[i][0] for i in range(len(dataset))]).permute(
                0, 2, 3, 1
            )
            targets = dataset.targets
        else:
            raise ValueError(f"Unsupported dataset: {name}")
        self.images = images
        self.targets = torch.as_tensor(targets, dtype=torch.long)
        self.device = torch.device(device)
        if self.device.type == "cuda" and name in ("cifar10", "cifar100"):
            self.images = self.images.pin_memory()
            self.targets = self.targets.pin_memory()
        self.size = SPECS[name]["size"]
        self.mean = torch.tensor(SPECS[name]["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(SPECS[name]["std"], device=device).view(1, 3, 1, 1)
        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device).manual_seed(seed)
        self.tiny_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(SPECS[name]["mean"], SPECS[name]["std"]),
            ]
        )

    def __len__(self):
        return len(self.targets)

    def batch(self, ids, augment=True):
        ids_cpu = (
            ids.detach().cpu()
            if isinstance(ids, torch.Tensor)
            else torch.as_tensor(ids)
        )
        if self.name == "tiny-imagenet" and augment:
            x = torch.stack(
                [
                    self.tiny_transform(Image.fromarray(self.images[int(i)].numpy()))
                    for i in ids_cpu
                ]
            )
            y = self.targets[ids_cpu].to(self.device)
            return x.to(self.device), y, ids_cpu.long()
        x = self.images[ids_cpu].to(self.device, non_blocking=True)
        y = self.targets[ids_cpu].to(self.device, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255)
        if augment:
            x = F.pad(x, (4, 4, 4, 4), mode="reflect")
            batch, channels, _, padded_width = x.shape
            oy = torch.randint(
                0, 9, (batch,), generator=self.generator, device=self.device
            )
            ox = torch.randint(
                0, 9, (batch,), generator=self.generator, device=self.device
            )
            rows = (
                oy[:, None, None, None]
                + torch.arange(self.size, device=self.device)[None, None, :, None]
            )
            rows = rows.expand(batch, channels, self.size, padded_width)
            x = torch.gather(x, 2, rows)
            cols = (
                ox[:, None, None, None]
                + torch.arange(self.size, device=self.device)[None, None, None, :]
            )
            cols = cols.expand(batch, channels, self.size, self.size)
            x = torch.gather(x, 3, cols)
            flip = torch.rand(batch, generator=self.generator, device=self.device) < 0.5
            x[flip] = x[flip].flip(-1)
        return x.sub_(self.mean).div_(self.std), y, ids_cpu.long()
