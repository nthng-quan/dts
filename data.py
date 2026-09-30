import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parent
SPECS = {
    "cifar10": {
        "classes": 10,
        "size": 32,
        "model": "r18",
        "train": 50000,
        "evaluation": 10000,
        "mean": (0.491400, 0.482158, 0.446531),
        "std": (0.247032, 0.243485, 0.261588),
    },
    "cifar100": {
        "classes": 100,
        "size": 32,
        "model": "r18",
        "train": 50000,
        "evaluation": 10000,
        "mean": (0.507075, 0.486549, 0.440918),
        "std": (0.267334, 0.256438, 0.276150),
    },
    "tiny-imagenet": {
        "classes": 200,
        "size": 64,
        "model": "r50",
        "train": 100000,
        "evaluation": 10000,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
    },
}


class TinyImageNet(Dataset):
    def __init__(self, root, train, transform):
        root = Path(root) / "tiny-imagenet-200"
        self.root = root
        self.archive_path = root.parent / "archives/tiny-imagenet-200.zip"
        self._archive = None
        self._archive_pid = None
        self.use_archive = self.archive_path.exists()
        if self.use_archive:
            with zipfile.ZipFile(self.archive_path) as z:
                wnids = z.read("tiny-imagenet-200/wnids.txt").decode()
                names = z.namelist()
                if not train:
                    annotations_text = z.read(
                        "tiny-imagenet-200/val/val_annotations.txt"
                    ).decode()
        else:
            wnids = (root / "wnids.txt").read_text()
        self.classes = sorted(wnids.split())
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.transform = transform
        if train:
            if self.use_archive:
                self.samples = [
                    (root.parent / name, self.class_to_idx[Path(name).parts[2]])
                    for name in sorted(names)
                    if name.startswith("tiny-imagenet-200/train/")
                    and name.endswith(".JPEG")
                    and Path(name).parent.name == "images"
                ]
            else:
                self.samples = [
                    (path, self.class_to_idx[name])
                    for name in self.classes
                    for path in sorted(
                        (root / "train" / name / "images").glob("*.JPEG")
                    )
                ]
        else:
            if not self.use_archive:
                annotations_text = (root / "val" / "val_annotations.txt").read_text()
            annotations = dict(
                line.split()[:2] for line in annotations_text.splitlines()
            )
            self.samples = [
                (root / "val" / "images" / name, self.class_to_idx[wnid])
                for name, wnid in sorted(annotations.items())
            ]
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if self.use_archive:
            if self._archive_pid != os.getpid():
                if self._archive is not None:
                    self._archive.close()
                self._archive = zipfile.ZipFile(self.archive_path)
                self._archive_pid = os.getpid()
            source = io.BytesIO(
                self._archive.read(path.relative_to(self.root.parent).as_posix())
            )
        else:
            source = path
        with Image.open(source) as image:
            image = image.convert("RGB")
        return self.transform(image), target


class Indexed(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = np.asarray(dataset.targets, dtype=np.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        return image, target, index


def build_dataset(name, root=ROOT / "data", train=True, augment=True):
    spec = SPECS[name]
    ops = []
    if name == "tiny-imagenet":
        if train and augment:
            ops += [transforms.RandomResizedCrop(64), transforms.RandomHorizontalFlip()]
        elif not train:
            ops += [transforms.Resize(73), transforms.CenterCrop(64)]
    elif train and augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(spec["size"], padding=4, padding_mode="reflect"),
        ]
    ops += [transforms.ToTensor(), transforms.Normalize(spec["mean"], spec["std"])]
    transform = transforms.Compose(ops)
    if name == "tiny-imagenet":
        base = TinyImageNet(root, train, transform)
    else:
        cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
        base = cls(str(root), train=train, download=False, transform=transform)
    expected = spec["train" if train else "evaluation"]
    if len(base) != expected:
        raise ValueError(f"{name}: expected {expected} samples, got {len(base)}")
    return Indexed(base)


def identity_hash(dataset):
    h = hashlib.sha256(dataset.targets.tobytes())
    if hasattr(dataset.dataset, "samples"):
        for path, _ in dataset.dataset.samples:
            h.update(f"{path.parent.parent.name}/{path.name}\n".encode())
    else:
        h.update(dataset.dataset.data.tobytes())
    return h.hexdigest()
