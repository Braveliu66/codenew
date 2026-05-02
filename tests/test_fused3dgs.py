from __future__ import annotations

import unittest

import torch

from fused3dgs.config import Fused3DGSConfig
from fused3dgs.cuda_backend import GaussianRasterizerBackend
from fused3dgs.modules.densification import MultiViewConsistencyDensification
from fused3dgs.optimizer import LMGaussianOptimizer
from fused3dgs.scene.gaussian_model import FusedGaussianModel
from fused3dgs.training_loop import FusedTrainingLoop


class FakeGaussianSet:
    num_gaussians = 2

    def __init__(self) -> None:
        self.densify_scores: torch.Tensor | None = None
        self.prune_scores: torch.Tensor | None = None

    def densify_from_vcd_scores(self, scores: torch.Tensor, *, grad_thresh: float, grad_abs_thresh: float) -> None:
        self.densify_scores = scores

    def prune_from_vcd_scores(self, scores: torch.Tensor, *, loss_thresh: float) -> None:
        self.prune_scores = scores


class FakeBackend(GaussianRasterizerBackend):
    def __init__(self) -> None:
        self.forward_calls: list[dict] = []
        self.lm_calls: list[str] = []

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        return {
            "rendered_images": [torch.zeros((1, 3, 3), dtype=torch.float32)],
            "projections": [
                {
                    "xy": torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
                    "radii": torch.ones(2, dtype=torch.float32),
                    "visibility": torch.tensor([True, True]),
                }
            ],
        }

    def eval_jtf_and_get_sparse_jacobian(self, **kwargs):
        self.lm_calls.append("jacobian")
        return {"cache": True}

    def sort_sparse_jacobians(self, jacobian_cache):
        self.lm_calls.append("sort")

    def calc_preconditioner(self, jacobian_cache):
        self.lm_calls.append("preconditioner")
        return {"preconditioner": True}

    def apply_lm_step(self, **kwargs):
        self.lm_calls.append("apply")
        return {"accepted": True}


class Fused3DGSTests(unittest.TestCase):
    def test_lm_schedule_uses_start_iteration_and_interval(self) -> None:
        config = Fused3DGSConfig.from_options(
            source_path="/tmp/source",
            model_path="/tmp/model",
            options={"lm_optimizer": {"start_iter": 3000, "interval": 200}},
        )

        self.assertFalse(config.should_run_lm(2999))
        self.assertTrue(config.should_run_lm(3000))
        self.assertFalse(config.should_run_lm(3100))
        self.assertTrue(config.should_run_lm(3200))

    def test_deblur_mlp_is_training_only_and_export_disabled(self) -> None:
        config = Fused3DGSConfig.from_options(
            source_path="/tmp/source",
            model_path="/tmp/model",
            options={"fused3dgs": {"use_deblur": "auto"}, "deblurring": {"start_iter": 10}},
        )
        model = FusedGaussianModel(config, blur_detected=True)
        xyz = torch.zeros((2, 3), dtype=torch.float32)

        model.train()
        self.assertIsNone(model.covariance_modulation(xyz, iteration=9))
        self.assertIsNotNone(model.covariance_modulation(xyz, iteration=10))
        self.assertIsNone(model.covariance_modulation(xyz, iteration=10, exporting=True))

        model.eval()
        self.assertIsNone(model.covariance_modulation(xyz, iteration=20))

    def test_deblur_modulates_covariance_with_bounded_delta(self) -> None:
        config = Fused3DGSConfig.from_options(
            source_path="/tmp/source",
            model_path="/tmp/model",
            options={"fused3dgs": {"use_deblur": True}, "deblurring": {"start_iter": 0}},
        )
        model = FusedGaussianModel(config, blur_detected=False)
        xyz = torch.zeros((2, 3), dtype=torch.float32)
        scaling = torch.ones((2, 3), dtype=torch.float32)
        rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

        model.train()
        mod_scaling, mod_rotation = model.modulate_covariance(xyz, scaling, rotation, iteration=0)

        self.assertEqual(mod_scaling.shape, scaling.shape)
        self.assertEqual(mod_rotation.shape, rotation.shape)
        self.assertTrue(torch.allclose(mod_rotation.norm(dim=-1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.all(mod_scaling > 0))

        same_scaling, same_rotation = model.modulate_covariance(xyz, scaling, rotation, iteration=0, exporting=True)
        self.assertTrue(torch.equal(same_scaling, scaling))
        self.assertTrue(torch.equal(same_rotation, rotation))

    def test_vcd_scores_gaussians_from_projected_loss_maps(self) -> None:
        vcd = MultiViewConsistencyDensification()
        rendered = [torch.zeros((1, 3, 3), dtype=torch.float32)]
        gt = [torch.zeros((1, 3, 3), dtype=torch.float32)]
        gt[0][0, 1, 1] = 1.0
        projections = [
            {
                "xy": torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
                "radii": torch.ones(2, dtype=torch.float32),
                "visibility": torch.tensor([True, True]),
            }
        ]

        scores = vcd.evaluate_gaussian_importance(vcd.build_loss_map(rendered, gt), projections)

        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_training_loop_schedules_sgd_vcd_and_lm(self) -> None:
        config = Fused3DGSConfig.from_options(
            source_path="/tmp/source",
            model_path="/tmp/model",
            options={
                "total_iterations": 5,
                "fused3dgs": {"use_deblur": False, "use_vcd": True, "use_lm_optimizer": True},
                "lm_optimizer": {"enabled": True, "start_iter": 2, "interval": 2},
                "vcd": {"enabled": True, "interval": 2, "densify_until_iter": 10},
            },
        )
        backend = FakeBackend()
        gaussians = FakeGaussianSet()
        lm = LMGaussianOptimizer(backend=backend, config=config.lm_optimizer)

        loop = FusedTrainingLoop(
            config=config,
            gaussians=gaussians,
            renderer=backend,
            batch_provider=lambda iteration: {"gt_images": [torch.zeros((1, 3, 3), dtype=torch.float32)]},
            sgd_step=lambda context: {},
            lm_optimizer=lm,
        )
        summary = loop.run()

        self.assertEqual(summary.actions(), ["sgd+vcd", "sgd", "lm", "sgd", "lm"])
        self.assertIsNotNone(gaussians.densify_scores)
        self.assertEqual(backend.lm_calls, ["jacobian", "sort", "preconditioner", "apply", "jacobian", "sort", "preconditioner", "apply"])

    def test_training_loop_passes_deblur_covariance_modulator_to_renderer(self) -> None:
        config = Fused3DGSConfig.from_options(
            source_path="/tmp/source",
            model_path="/tmp/model",
            options={
                "total_iterations": 2,
                "fused3dgs": {"use_deblur": "auto", "use_vcd": False, "use_lm_optimizer": False},
                "deblurring": {"start_iter": 1},
            },
        )
        backend = FakeBackend()
        deblur_model = FusedGaussianModel(config, blur_detected=True)
        loop = FusedTrainingLoop(
            config=config,
            gaussians=FakeGaussianSet(),
            renderer=backend,
            batch_provider=lambda iteration: {"gt_images": [torch.zeros((1, 3, 3), dtype=torch.float32)]},
            sgd_step=lambda context: {},
            deblur_model=deblur_model,
            blur_detected=True,
        )

        summary = loop.run()

        self.assertEqual(summary.actions(), ["sgd", "sgd"])
        self.assertIsNone(backend.forward_calls[0]["covariance_modulator"])
        self.assertIsNotNone(backend.forward_calls[1]["covariance_modulator"])


if __name__ == "__main__":
    unittest.main()
