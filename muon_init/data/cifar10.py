"""CIFAR-10 data loaders.

Provides two variants:
- flat: no augmentation, for MLP (images flattened in model forward)
- augmented: random crop + horizontal flip, for ResNet/ViT
"""

import torch
from torchvision import datasets, transforms


def get_cifar10_loaders(batch_size=128, augment=False, data_dir="./data_cache",
                        num_workers=2):
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(data_dir, train=True, download=True,
                                 transform=train_transform)
    val_set = datasets.CIFAR10(data_dir, train=False, download=True,
                               transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
