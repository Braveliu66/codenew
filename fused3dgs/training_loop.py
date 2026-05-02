from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from fused3dgs.config import Fused3DGSConfig
from fused3dgs.cuda_backend import GaussianRasterizerBackend
from fused3dgs.modules.densification import MultiViewConsistencyDensification
from fused3dgs.optimizer import LMGaussianOptimizer


class BatchProvider(Protocol):
    def __call__(self, iteration: int) -> dict[str, Any]:
        ...


class SGDStep(Protocol):
    def __call__(self, context: "TrainingStepContext") -> dict[str, Any] | None:
        ...


@dataclass(frozen=True)
class TrainingStepContext:
    iteration: int
    gaussians: Any
    batch: dict[str, Any]
    render_pkg: Any
    deblur_active: bool


@dataclass
class TrainingEvent:
    iteration: int
    action: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRunSummary:
    events: list[TrainingEvent] = field(default_factory=list)

    def actions(self) -> list[str]:
        return [event.action for event in self.events]


class FusedTrainingLoop:
    """Orchestrates SGD, VCD, Deblurring, and interval LM steps.

    This class intentionally contains orchestration only. Real rendering,
    gradients, optimizer updates, and CUDA kernels are injected through the
    backend and callbacks so the platform can test scheduling without faking
    model artifacts.
    """

    def __init__(
        self,
        *,
        config: Fused3DGSConfig,
        gaussians: Any,
        renderer: GaussianRasterizerBackend,
        batch_provider: BatchProvider,
        sgd_step: SGDStep,
        lm_optimizer: LMGaussianOptimizer | None = None,
        vcd: MultiViewConsistencyDensification | None = None,
        deblur_model: Any | None = None,
        blur_detected: bool = False,
    ) -> None:
        self.config = config
        self.gaussians = gaussians
        self.renderer = renderer
        self.batch_provider = batch_provider
        self.sgd_step = sgd_step
        self.lm_optimizer = lm_optimizer
        self.vcd = vcd or MultiViewConsistencyDensification(
            loss_thresh=config.vcd.loss_thresh,
            grad_thresh=config.vcd.grad_thresh,
            grad_abs_thresh=config.vcd.grad_abs_thresh,
        )
        self.deblur_model = deblur_model
        self.blur_detected = blur_detected

    def run(self, *, start_iteration: int = 0, total_iterations: int | None = None) -> TrainingRunSummary:
        end_iteration = total_iterations if total_iterations is not None else self.config.total_iterations
        summary = TrainingRunSummary()
        for iteration in range(start_iteration, end_iteration):
            summary.events.append(self.step(iteration))
        return summary

    def step(self, iteration: int) -> TrainingEvent:
        batch = self.batch_provider(iteration)
        if self.config.should_run_lm(iteration):
            return self._lm_step(iteration, batch)
        return self._sgd_step(iteration, batch)

    def _lm_step(self, iteration: int, batch: dict[str, Any]) -> TrainingEvent:
        if self.lm_optimizer is None:
            raise RuntimeError("LM optimizer is enabled by schedule but no LMGaussianOptimizer was provided")
        result = self.lm_optimizer.step(
            gaussians=self.gaussians,
            viewpoints=batch.get("viewpoints"),
            pipe=batch.get("pipe"),
            background=batch.get("background"),
            iteration=iteration,
        )
        return TrainingEvent(iteration=iteration, action="lm", details={"result": result})

    def _sgd_step(self, iteration: int, batch: dict[str, Any]) -> TrainingEvent:
        deblur_active = self._deblur_active(iteration)
        render_pkg = self.renderer.forward(
            gaussians=self.gaussians,
            viewpoints=batch.get("viewpoints"),
            pipe=batch.get("pipe"),
            background=batch.get("background"),
            iteration=iteration,
            covariance_modulator=self._covariance_modulator(iteration) if deblur_active else None,
        )
        sgd_result = self.sgd_step(
            TrainingStepContext(
                iteration=iteration,
                gaussians=self.gaussians,
                batch=batch,
                render_pkg=render_pkg,
                deblur_active=deblur_active,
            )
        ) or {}
        details: dict[str, Any] = {"deblur_active": deblur_active, "sgd_result": sgd_result}
        if self.config.should_run_vcd(iteration):
            scores = self._run_vcd(batch=batch, render_pkg=render_pkg, sgd_result=sgd_result)
            details["vcd_scores_shape"] = tuple(scores.shape)
            return TrainingEvent(iteration=iteration, action="sgd+vcd", details=details)
        return TrainingEvent(iteration=iteration, action="sgd", details=details)

    def _deblur_active(self, iteration: int) -> bool:
        active = self.config.should_use_deblur(
            blur_detected=self.blur_detected,
            training=True,
            iteration=iteration,
            exporting=False,
        )
        if active and self.deblur_model is None:
            raise RuntimeError("Deblurring is enabled but no deblur model was provided")
        if active and hasattr(self.deblur_model, "train"):
            self.deblur_model.train()
        return active

    def _covariance_modulator(self, iteration: int) -> Callable[..., Any]:
        def modulate(*, xyz: Any, scaling: Any, rotation: Any) -> Any:
            if not hasattr(self.deblur_model, "modulate_covariance"):
                raise RuntimeError("deblur model does not expose modulate_covariance")
            return self.deblur_model.modulate_covariance(
                xyz,
                scaling,
                rotation,
                iteration=iteration,
                exporting=False,
            )

        return modulate

    def _run_vcd(self, *, batch: dict[str, Any], render_pkg: Any, sgd_result: dict[str, Any]) -> Any:
        rendered_images = self._first_present("rendered_images", sgd_result, render_pkg, batch)
        gt_images = self._first_present("gt_images", batch, sgd_result)
        projections = self._first_present("projections", sgd_result, render_pkg, batch)
        if rendered_images is None or gt_images is None or projections is None:
            raise RuntimeError("VCD requires rendered_images, gt_images, and projections from the training step")
        return self.vcd.densify_and_prune_from_views(
            self.gaussians,
            rendered_images=list(rendered_images),
            gt_images=list(gt_images),
            projections=list(projections),
            num_gaussians=self._num_gaussians(),
        )

    def _num_gaussians(self) -> int | None:
        if hasattr(self.gaussians, "num_gaussians"):
            value = self.gaussians.num_gaussians
            return int(value() if callable(value) else value)
        if hasattr(self.gaussians, "__len__"):
            return len(self.gaussians)
        return None

    def _first_present(self, key: str, *sources: Any) -> Any:
        for source in sources:
            if isinstance(source, dict) and key in source:
                value = source[key]
                if value is not None:
                    return value
            if hasattr(source, key):
                value = getattr(source, key)
                if value is not None:
                    return value
        return None
