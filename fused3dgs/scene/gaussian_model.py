from __future__ import annotations

import torch
from torch import nn

from fused3dgs.config import Fused3DGSConfig


class DeblurMLP(nn.Module):
    """Training-only covariance modulation MLP adapted from Deblurring-3DGS."""

    def __init__(self, d_in: int = 3, d_hidden: int = 64, d_out: int = 7, layers: int = 3) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("DeblurMLP requires at least two layers")
        modules: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.ReLU()]
        for _ in range(max(layers - 2, 0)):
            modules.extend([nn.Linear(d_hidden, d_hidden), nn.ReLU()])
        modules.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*modules)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(xyz)
        scale_mod = out[:, :3]
        rot_mod = out[:, 3:]
        return scale_mod, rot_mod


class FusedGaussianModel(nn.Module):
    """Minimal Gaussian model integration surface for the fused training loop."""

    def __init__(self, config: Fused3DGSConfig, *, blur_detected: bool = False) -> None:
        super().__init__()
        self.config = config
        self.blur_detected = blur_detected
        self.deblur_mlp = DeblurMLP(
            d_hidden=config.deblurring.mlp_hidden,
            layers=config.deblurring.mlp_layers,
        )

    def should_apply_deblur(self, *, iteration: int, exporting: bool = False) -> bool:
        return self.config.should_use_deblur(
            blur_detected=self.blur_detected,
            training=self.training,
            iteration=iteration,
            exporting=exporting,
        )

    def covariance_modulation(
        self,
        xyz: torch.Tensor,
        *,
        iteration: int,
        exporting: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.should_apply_deblur(iteration=iteration, exporting=exporting):
            return None
        return self.deblur_mlp(xyz)

    def modulate_covariance(
        self,
        xyz: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        *,
        iteration: int,
        exporting: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        modulation = self.covariance_modulation(xyz, iteration=iteration, exporting=exporting)
        if modulation is None:
            return scaling, rotation

        scale_mod, rot_mod = modulation
        scale_delta = torch.tanh(scale_mod) * self.config.deblurring.max_scale_delta
        rotation_delta = torch.tanh(rot_mod) * self.config.deblurring.max_rotation_delta
        modulated_scaling = scaling * torch.exp(scale_delta)
        modulated_rotation = torch.nn.functional.normalize(rotation + rotation_delta, dim=-1)
        return modulated_scaling, modulated_rotation
