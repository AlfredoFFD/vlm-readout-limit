"""Mock API client for end-to-end tests of the experiment runner.

The mock client implements the `APIClient` interface but does not call any
external service. It deterministically labels each trial based on the trial_id,
or by inspecting the rendered pixel difference of the two images supplied
(so it can simulate either a "perfect" or a "noisy" model).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Sequence

from .base import APIClient


class MockClient(APIClient):
    provider = "mock"

    def __init__(self, mode: str = "deterministic_yes", model: str | None = None, **kwargs):
        """
        mode:
          - "deterministic_yes"  : always answers YES (constant model)
          - "deterministic_no"   : always answers NO
          - "perfect"            : checks the actual pixel diff and answers YES iff different
          - "chance"             : pseudo-random per trial (hash of image paths)
        """
        super().__init__(model=model or f"mock-{mode}", **kwargs)
        self.mode = mode

    def _answer(self, images: Sequence[Path]) -> str:
        if self.mode == "deterministic_yes":
            return "YES"
        if self.mode == "deterministic_no":
            return "NO"
        if self.mode == "chance":
            h = hashlib.sha256("|".join(str(p) for p in images).encode()).digest()[0]
            return "YES" if h % 2 == 0 else "NO"
        if self.mode == "perfect":
            # Look at the actual image bytes to detect "change".
            from PIL import Image
            import numpy as np
            arrs = [np.asarray(Image.open(p).convert("RGB")) for p in images]
            if len(arrs) == 1:
                # Combined image (sample | divider | test) -- the two halves are the
                # same width as a single canvas. We strip the divider by taking the
                # leftmost CANVAS_W columns and the rightmost CANVAS_W columns.
                from vwm.stimuli.geometry import CANVAS_W
                a = arrs[0]
                left = a[:, :CANVAS_W]
                right = a[:, -CANVAS_W:]
                diff = float(np.abs(left.astype("int32") - right.astype("int32")).mean())
                return "YES" if diff > 0.5 else "NO"
            if len(arrs) >= 2:
                diff = float(np.abs(arrs[0].astype("int32") - arrs[1].astype("int32")).mean())
                return "YES" if diff > 0.5 else "NO"
        return "NO"

    def _single_turn(self, images, prompt, system, max_tokens=8):
        time.sleep(0.001)
        return self._answer(images), {"prompt": 0, "completion": 1}

    def _two_turn(self, sample_image, sample_prompt, test_image, test_prompt, system, max_tokens=8):
        time.sleep(0.001)
        return self._answer([sample_image, test_image]), {"prompt": 0, "completion": 1}
