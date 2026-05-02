from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SpeedyPrunePolicy:
    opacity_threshold: float = 0.005
    max_screen_size: float | None = None

    def keep_mask(self, opacity: torch.Tensor, screen_size: torch.Tensor | None = None) -> torch.Tensor:
        mask = opacity.reshape(-1) >= self.opacity_threshold
        if self.max_screen_size is not None and screen_size is not None:
            mask &= screen_size.reshape(-1) <= self.max_screen_size
        return mask
