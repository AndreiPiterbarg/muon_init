"""Data loader factory."""

from data.cifar10 import get_cifar10_loaders
from data.openwebtext import get_openwebtext_loaders
from data.wikitext import get_wikitext103_loaders


def build_dataloaders(config):
    """Build train/val data loaders from a config dict.

    Config must have a 'name' key. Remaining keys are passed as kwargs.
    """
    name = config["name"]
    kwargs = {k: v for k, v in config.items() if k != "name"}

    if name == "cifar10":
        return get_cifar10_loaders(**kwargs)
    elif name == "cifar10_augmented":
        return get_cifar10_loaders(augment=True, **kwargs)
    elif name == "openwebtext":
        return get_openwebtext_loaders(**kwargs)
    elif name == "wikitext103":
        return get_wikitext103_loaders(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
