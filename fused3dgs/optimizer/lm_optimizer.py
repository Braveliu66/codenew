from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fused3dgs.config import LMOptimizerConfig
from fused3dgs.cuda_backend import GaussianRasterizerBackend


@dataclass
class LMGaussianOptimizer:
    backend: GaussianRasterizerBackend
    config: LMOptimizerConfig

    def should_run(self, iteration: int) -> bool:
        return self.config.should_run(iteration)

    def step(self, *, gaussians: Any, viewpoints: Any, pipe: Any, background: Any, iteration: int) -> Any:
        if not self.should_run(iteration):
            raise RuntimeError(f"LM step was called at iteration {iteration}, outside the configured schedule")
        render_pkg = self.backend.forward(gaussians=gaussians, viewpoints=viewpoints, pipe=pipe, background=background)
        jacobian_cache = self.backend.eval_jtf_and_get_sparse_jacobian(render_pkg=render_pkg, gaussians=gaussians, viewpoints=viewpoints)
        self.backend.sort_sparse_jacobians(jacobian_cache)
        preconditioner = self.backend.calc_preconditioner(jacobian_cache)
        return self.backend.apply_lm_step(
            jacobian_cache=jacobian_cache,
            preconditioner=preconditioner,
            pcg_rtol=self.config.pcg_rtol,
            pcg_max_iter=self.config.pcg_max_iter,
            trust_region_radius=self.config.trust_region_radius,
        )
