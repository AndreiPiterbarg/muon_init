"""Model factory."""

from models.mlp import DeepMLP
from models.resnet import SmallResNet18
from models.gpt import GPT
from models.vit import ViTTiny


def build_model(config):
    """Build a model from a config dict.

    Config must have a 'name' key. Remaining keys are passed as kwargs.
    """
    name = config["name"]
    kwargs = {k: v for k, v in config.items() if k != "name"}

    if name == "deep_mlp":
        return DeepMLP(**kwargs)
    elif name == "small_resnet18":
        return SmallResNet18(**kwargs)
    elif name == "nanogpt":
        return GPT(**kwargs)
    elif name == "deep_narrow_gpt":
        return GPT(post_norm=True, **kwargs)
    elif name == "vit_tiny":
        return ViTTiny(**kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")
