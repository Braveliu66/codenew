from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


class GaussianSet(Protocol):
    def densify_from_vcd_scores(self, scores: torch.Tensor, *, grad_thresh: float, grad_abs_thresh: float) -> None:
        ...

    def prune_from_vcd_scores(self, scores: torch.Tensor, *, loss_thresh: float) -> None:
        ...


@dataclass
class MultiViewConsistencyDensification:
    """FastGS VCD integration boundary.

    The real projection and contribution scoring logic is adapted from the
    FastGS repository. This class defines the stable call surface used by the
    fused training loop and refuses to silently densify without scores.
    """

    loss_thresh: float = 0.1
    grad_thresh: float = 0.0002
    grad_abs_thresh: float = 0.001

    def build_loss_map(self, rendered_images: list[torch.Tensor], gt_images: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(rendered_images) != len(gt_images):
            raise ValueError("rendered_images and gt_images must have the same length")
        loss_maps: list[torch.Tensor] = []
        for rendered, gt in zip(rendered_images, gt_images):
            loss = torch.abs(rendered - gt).mean(dim=0)
            loss_maps.append((loss - loss.min()) / (loss.max() - loss.min() + 1e-8))
        return loss_maps

    def evaluate_gaussian_importance(
        self,
        loss_maps: list[torch.Tensor],
        projections: list[dict[str, Any]],
        *,
        num_gaussians: int | None = None,
    ) -> torch.Tensor:
        if len(loss_maps) != len(projections):
            raise ValueError("loss_maps and projections must have the same length")
        if not loss_maps:
            raise ValueError("at least one loss map is required")

        first_xy = self._projection_xy(projections[0])
        count = int(num_gaussians or first_xy.shape[0])
        device = first_xy.device
        scores = torch.zeros(count, dtype=torch.float32, device=device)
        visible_counts = torch.zeros(count, dtype=torch.float32, device=device)

        for loss_map, projection in zip(loss_maps, projections):
            xy = self._projection_xy(projection).to(device=device)
            if xy.shape[0] != count:
                raise ValueError("all projection tensors must contain the same gaussian count")
            if loss_map.ndim == 3:
                loss_map = loss_map.mean(dim=0)
            if loss_map.ndim != 2:
                raise ValueError("loss maps must be HxW or CxHxW tensors")
            loss_map = loss_map.to(device=device, dtype=torch.float32)
            height, width = loss_map.shape
            x = xy[:, 0]
            y = xy[:, 1]
            visible = self._projection_visibility(projection, x=x, y=y, width=width, height=height).to(device=device)
            x_index = x.round().long().clamp(0, width - 1)
            y_index = y.round().long().clamp(0, height - 1)
            sampled_loss = loss_map[y_index, x_index]
            radii = self._projection_radii(projection, count=count, device=device)
            footprint_weight = torch.clamp(radii, min=1.0).square()
            scores += torch.where(visible, sampled_loss * footprint_weight, torch.zeros_like(sampled_loss))
            visible_counts += visible.to(dtype=torch.float32)

        return scores / visible_counts.clamp_min(1.0)

    def densify_and_prune_from_views(
        self,
        gaussians: GaussianSet,
        *,
        rendered_images: list[torch.Tensor],
        gt_images: list[torch.Tensor],
        projections: list[dict[str, Any]],
        num_gaussians: int | None = None,
    ) -> torch.Tensor:
        loss_maps = self.build_loss_map(rendered_images, gt_images)
        scores = self.evaluate_gaussian_importance(loss_maps, projections, num_gaussians=num_gaussians)
        self.densify_and_prune(gaussians, scores)
        return scores

    def densify_and_prune(self, gaussians: GaussianSet, scores: torch.Tensor | None) -> None:
        if scores is None:
            raise RuntimeError("FastGS VCD scores are required; refusing to run placeholder densification")
        gaussians.densify_from_vcd_scores(scores, grad_thresh=self.grad_thresh, grad_abs_thresh=self.grad_abs_thresh)
        gaussians.prune_from_vcd_scores(scores, loss_thresh=self.loss_thresh)

    def _projection_xy(self, projection: dict[str, Any]) -> torch.Tensor:
        value = projection.get("xy", projection.get("means2d"))
        if value is None:
            raise ValueError("projection must include xy or means2d")
        tensor = torch.as_tensor(value)
        if tensor.ndim != 2 or tensor.shape[1] != 2:
            raise ValueError("projection xy must have shape Nx2")
        return tensor

    def _projection_visibility(
        self,
        projection: dict[str, Any],
        *,
        x: torch.Tensor,
        y: torch.Tensor,
        width: int,
        height: int,
    ) -> torch.Tensor:
        inside = torch.isfinite(x) & torch.isfinite(y) & (x >= 0) & (y >= 0) & (x < width) & (y < height)
        explicit = projection.get("visibility", projection.get("visible"))
        if explicit is None:
            return inside
        return inside & torch.as_tensor(explicit, dtype=torch.bool, device=x.device)

    def _projection_radii(self, projection: dict[str, Any], *, count: int, device: torch.device) -> torch.Tensor:
        value = projection.get("radii", projection.get("radius"))
        if value is None:
            return torch.ones(count, dtype=torch.float32, device=device)
        radii = torch.as_tensor(value, dtype=torch.float32, device=device)
        if radii.ndim == 0:
            return radii.expand(count)
        if radii.shape[0] != count:
            raise ValueError("projection radii must match gaussian count")
        return radii
