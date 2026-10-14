"""CCOM-6070 final-project deliverable.

A trained-from-scratch Siamese-CNN change detector plus classical (pixel-diff) and
random baselines, trained and evaluated on the Luck-Vogel stimuli we generate for
the preprint. Shares the same stimuli pipeline so the course and preprint stay in
lock-step.
"""

from .dataset import LuckVogelDataset, split_trials_by_set_size
from .model import SiameseCNN, BasicBlock
from .baselines import (
    pixel_diff_change_detector,
    random_change_detector,
    evaluate_detector,
)
from .train import train_siamese, evaluate_siamese

__all__ = [
    "LuckVogelDataset",
    "split_trials_by_set_size",
    "SiameseCNN",
    "BasicBlock",
    "pixel_diff_change_detector",
    "random_change_detector",
    "evaluate_detector",
    "train_siamese",
    "evaluate_siamese",
]
