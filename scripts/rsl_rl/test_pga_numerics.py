from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch


RSL_RL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(RSL_RL_ROOT))

from amtl.ppo import (  # noqa: E402
    HIERARCHICAL_3_PGA_GROUPS,
    _stable_symmetric_eigh,
    aggregate_hierarchical_3_pga,
)


class StableSymmetricEighTests(unittest.TestCase):
    def test_symmetrizes_input(self) -> None:
        matrix = torch.tensor([[2.0, 1.0002], [0.9998, 3.0]], dtype=torch.float32)

        eigenvalues, eigenvectors = _stable_symmetric_eigh(matrix)

        reconstructed = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.T
        expected = 0.5 * (matrix + matrix.T)
        torch.testing.assert_close(reconstructed, expected)

    def test_retries_in_cpu_float64(self) -> None:
        matrix = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        original_eigh = torch.linalg.eigh
        attempted_dtypes: list[torch.dtype] = []

        def fail_float32_once(value: torch.Tensor):
            attempted_dtypes.append(value.dtype)
            if value.dtype == torch.float32:
                raise RuntimeError("simulated primary solver failure")
            return original_eigh(value)

        with mock.patch("torch.linalg.eigh", side_effect=fail_float32_once):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                eigenvalues, eigenvectors = _stable_symmetric_eigh(matrix)

        self.assertEqual(attempted_dtypes, [torch.float32, torch.float64])
        self.assertEqual(eigenvalues.dtype, matrix.dtype)
        self.assertEqual(eigenvectors.dtype, matrix.dtype)
        self.assertEqual(eigenvalues.device, matrix.device)
        self.assertEqual(eigenvectors.device, matrix.device)
        self.assertIn("CPU float64 retry", str(caught[0].message))
        torch.testing.assert_close(eigenvalues, torch.tensor([1.0, 3.0]))


class HierarchicalThreePgaTests(unittest.TestCase):
    @staticmethod
    def objective_names() -> list[str]:
        return [
            objective_name
            for _, group_objective_names, _ in HIERARCHICAL_3_PGA_GROUPS
            for objective_name in group_objective_names
        ]

    def test_outer_inputs_match_configured_group_budgets(self) -> None:
        torch.manual_seed(7)
        objective_names = self.objective_names()
        gradients = torch.randn(len(objective_names), 64, dtype=torch.float64)
        gradients[objective_names.index("action_rate_l2")] *= 1.0e6

        result = aggregate_hierarchical_3_pga(
            gradients,
            objective_names,
            rank=16,
            direction_weighting="factor_strength",
        )

        expected_budgets = gradients.new_tensor([0.70, 0.20, 0.10])
        torch.testing.assert_close(
            result.weighted_group_grads.norm(dim=1),
            expected_budgets,
        )
        torch.testing.assert_close(
            gradients.T @ result.objective_weights,
            result.aggregate_grad,
        )

    def test_action_rate_scale_does_not_change_hierarchical_update(self) -> None:
        torch.manual_seed(11)
        objective_names = self.objective_names()
        gradients = torch.randn(len(objective_names), 64, dtype=torch.float64)
        scaled_gradients = gradients.clone()
        scaled_gradients[objective_names.index("action_rate_l2")] *= 1.0e6

        common_args = {
            "rank": 16,
            "direction_weighting": "factor_strength",
        }
        baseline = aggregate_hierarchical_3_pga(
            gradients,
            objective_names,
            **common_args,
        )
        scaled = aggregate_hierarchical_3_pga(
            scaled_gradients,
            objective_names,
            **common_args,
        )

        torch.testing.assert_close(scaled.aggregate_grad, baseline.aggregate_grad)

    def test_inactive_safety_group_contributes_zero(self) -> None:
        torch.manual_seed(19)
        objective_names = self.objective_names()
        gradients = torch.randn(len(objective_names), 64, dtype=torch.float64)
        gradients[objective_names.index("joint_limit")].zero_()
        gradients[objective_names.index("undesired_contacts")].zero_()

        result = aggregate_hierarchical_3_pga(
            gradients,
            objective_names,
            rank=16,
            direction_weighting="factor_strength",
        )

        self.assertEqual(result.weighted_group_grads[1].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(result.aggregate_grad).all())
        torch.testing.assert_close(
            gradients.T @ result.objective_weights,
            result.aggregate_grad,
        )


if __name__ == "__main__":
    unittest.main()
