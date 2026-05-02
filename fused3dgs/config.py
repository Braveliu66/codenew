from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FusedModuleConfig:
    use_deblur: bool | str = "auto"
    use_vcd: bool = True
    use_lm_optimizer: bool = True


@dataclass(frozen=True)
class DeblurringConfig:
    start_iter: int = 0
    mlp_hidden: int = 64
    mlp_layers: int = 3
    max_scale_delta: float = 0.25
    max_rotation_delta: float = 0.05


@dataclass(frozen=True)
class LMOptimizerConfig:
    enabled: bool = True
    start_iter: int = 3000
    interval: int = 200
    pcg_rtol: float = 0.05
    pcg_max_iter: int = 8
    trust_region_radius: float = 1e-3
    min_trust_region_radius: float = 1e-4
    max_trust_region_radius: float = 1e-2

    def should_run(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        if iteration < self.start_iter:
            return False
        return self.interval > 0 and iteration % self.interval == 0


@dataclass(frozen=True)
class VCDConfig:
    enabled: bool = True
    loss_thresh: float = 0.1
    grad_thresh: float = 0.0002
    grad_abs_thresh: float = 0.001
    densify_until_iter: int = 15000
    interval: int = 100


@dataclass(frozen=True)
class OutputConfig:
    spz: bool = True
    lod: bool = True
    metrics: bool = True


@dataclass(frozen=True)
class Fused3DGSConfig:
    source_path: str
    model_path: str
    fused3dgs: FusedModuleConfig = field(default_factory=FusedModuleConfig)
    deblurring: DeblurringConfig = field(default_factory=DeblurringConfig)
    lm_optimizer: LMOptimizerConfig = field(default_factory=LMOptimizerConfig)
    vcd: VCDConfig = field(default_factory=VCDConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    lod_targets: dict[int, int] = field(default_factory=lambda: {0: 1_000_000, 1: 500_000, 2: 200_000, 3: 50_000})
    total_iterations: int = 30000

    @classmethod
    def from_options(
        cls,
        *,
        source_path: str,
        model_path: str,
        options: dict[str, Any] | None = None,
    ) -> "Fused3DGSConfig":
        data = dict(options or {})
        return cls(
            source_path=source_path,
            model_path=model_path,
            fused3dgs=FusedModuleConfig(**section(data, "fused3dgs")),
            deblurring=DeblurringConfig(**section(data, "deblurring")),
            lm_optimizer=LMOptimizerConfig(**section(data, "lm_optimizer")),
            vcd=VCDConfig(**section(data, "vcd")),
            outputs=OutputConfig(**section(data, "outputs")),
            lod_targets=normalize_lod_targets(data.get("lod_targets")),
            total_iterations=int(data.get("total_iterations") or 30000),
        )

    def should_run_lm(self, iteration: int) -> bool:
        return self.fused3dgs.use_lm_optimizer and self.lm_optimizer.should_run(iteration)

    def should_run_vcd(self, iteration: int) -> bool:
        if not self.fused3dgs.use_vcd or not self.vcd.enabled:
            return False
        if iteration > self.vcd.densify_until_iter:
            return False
        return self.vcd.interval > 0 and iteration % self.vcd.interval == 0

    def should_use_deblur(self, *, blur_detected: bool, training: bool, iteration: int, exporting: bool = False) -> bool:
        if exporting or not training:
            return False
        configured = self.fused3dgs.use_deblur
        if isinstance(configured, bool):
            enabled = configured
        else:
            enabled = str(configured).lower() == "auto" and blur_detected
        return enabled and iteration >= self.deblurring.start_iter

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return dict(value) if isinstance(value, dict) else {}


def normalize_lod_targets(raw: object) -> dict[int, int]:
    if not isinstance(raw, dict):
        return {0: 1_000_000, 1: 500_000, 2: 200_000, 3: 50_000}
    targets: dict[int, int] = {}
    for key, value in raw.items():
        lod = int(key)
        if lod in {0, 1, 2, 3}:
            targets[lod] = int(value)
    for lod, fallback in {0: 1_000_000, 1: 500_000, 2: 200_000, 3: 50_000}.items():
        targets.setdefault(lod, fallback)
    return targets
