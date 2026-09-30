import argparse
import json
import urllib.request
from pathlib import Path

from torchvision import datasets, transforms

from data import TinyImageNet

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def prepare_cifar(dataset_name, data_root):
    dataset_class = datasets.CIFAR10 if dataset_name == "cifar10" else datasets.CIFAR100
    training = dataset_class(data_root, train=True, download=True)
    evaluation = dataset_class(data_root, train=False, download=True)
    return {"training_samples": len(training), "evaluation_samples": len(evaluation)}


def prepare_tiny_imagenet(data_root, url):
    archive = data_root / "archives" / "tiny-imagenet-200.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        urllib.request.urlretrieve(url, archive)
    identity = transforms.PILToTensor()
    training = TinyImageNet(data_root, train=True, transform=identity)
    evaluation = TinyImageNet(data_root, train=False, transform=identity)
    return {
        "archive": str(archive),
        "training_samples": len(training),
        "evaluation_samples": len(evaluation),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["cifar10", "cifar100", "tiny-imagenet"], required=True
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tiny-imagenet-url", default=TINY_IMAGENET_URL)
    args = parser.parse_args()
    args.data_root.mkdir(parents=True, exist_ok=True)
    if args.dataset == "tiny-imagenet":
        summary = prepare_tiny_imagenet(args.data_root, args.tiny_imagenet_url)
    else:
        summary = prepare_cifar(args.dataset, str(args.data_root))
    print(json.dumps({"dataset": args.dataset, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
