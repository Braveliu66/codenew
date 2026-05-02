from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any


class GaussianRasterizerBackend(ABC):
    @abstractmethod
    def forward(self, **kwargs: Any) -> Any:
        raise NotImplementedError

    def eval_jtf_and_get_sparse_jacobian(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This backend does not provide 3DGS-LM sparse Jacobian kernels")

    def sort_sparse_jacobians(self, jacobian_cache: Any) -> Any:
        raise NotImplementedError("This backend does not provide 3DGS-LM sparse Jacobian sorting")

    def calc_preconditioner(self, jacobian_cache: Any) -> Any:
        raise NotImplementedError("This backend does not provide 3DGS-LM preconditioner kernels")

    def apply_lm_step(self, **kwargs: Any) -> Any:
        raise NotImplementedError("This backend does not provide 3DGS-LM JTJ/PCG kernels")


class FasterGSBackend(GaussianRasterizerBackend):
    """Dependency-injected wrapper around Faster-GS rasterization."""

    def __init__(self, module_name: str = "diff_gaussian_rasterization") -> None:
        self.module_name = module_name
        self.module = importlib.import_module(module_name)

    def forward(self, **kwargs: Any) -> Any:
        rasterize = getattr(self.module, "rasterize_gaussians", None) or getattr(self.module, "GaussianRasterizer", None)
        if rasterize is None:
            raise RuntimeError(f"{self.module_name} does not expose a Faster-GS rasterization entrypoint")
        if isinstance(rasterize, type):
            return rasterize(**kwargs)
        return rasterize(**kwargs)


class LMBackend(GaussianRasterizerBackend):
    """Dependency-injected wrapper around 3DGS-LM-specific kernels."""

    def __init__(self, module_name: str = "diff_gaussian_rasterization") -> None:
        self.module_name = module_name
        self.module = importlib.import_module(module_name)

    def forward(self, **kwargs: Any) -> Any:
        rasterize = getattr(self.module, "rasterize_gaussians", None)
        if rasterize is None:
            raise RuntimeError(f"{self.module_name} does not expose a rasterization entrypoint")
        return rasterize(**kwargs)

    def eval_jtf_and_get_sparse_jacobian(self, **kwargs: Any) -> Any:
        return self._call("eval_jtf_and_get_sparse_jacobian", **kwargs)

    def sort_sparse_jacobians(self, jacobian_cache: Any) -> Any:
        return self._call("sort_sparse_jacobians", jacobian_cache)

    def calc_preconditioner(self, jacobian_cache: Any) -> Any:
        return self._call("calc_preconditioner", jacobian_cache)

    def apply_lm_step(self, **kwargs: Any) -> Any:
        return self._call("apply_lm_step", **kwargs)

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        fn = getattr(self.module, name, None)
        if fn is None:
            raise RuntimeError(f"{self.module_name} does not expose {name}; check the 3DGS-LM rasterizer build")
        return fn(*args, **kwargs)


class FusedBackend(GaussianRasterizerBackend):
    """Future combined Faster-GS + 3DGS-LM CUDA backend."""

    def forward(self, **_: Any) -> Any:
        raise NotImplementedError("FusedBackend is reserved for a future merged CUDA extension")
