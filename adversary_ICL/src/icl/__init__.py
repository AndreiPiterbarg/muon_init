"""
ICL (In-Context Learning) module.

Based on: https://github.com/dtsip/in-context-learning
Paper: "What Can Transformers Learn In-Context?" (Garg et al., 2022)
"""

from .models import build_model, TransformerModel
from .tasks import get_task_sampler
from .samplers import get_data_sampler
from .curriculum import Curriculum
