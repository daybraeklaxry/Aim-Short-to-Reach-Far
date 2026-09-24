"""CPU tests of retrieval and CEM; requires torch, no benchmark assets."""
import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "torch is not installed; native kernel tests skipped")
class NativeKernelTests(unittest.TestCase):
    def test_bank_excludes_held_out_episodes_and_incomplete_options(self):
        import torch
        from anchored_planning.observation_bank import ObservationBank
        latents = np.random.default_rng(17).normal(size=(18, 2)).astype(np.float32)
        bank = ObservationBank(dict(lengths=np.array([9, 9]), offsets=np.array([0, 9])),
            latents, np.array([0]), 5, np.ones(3, np.float32), torch.device("cpu"))
        valid = bank.for_delta(5)[0]
        np.testing.assert_array_equal(valid, [0, 1, 2, 3])
        indices, distance = bank.nearest_topk(latents[[2]], latents[[7]], 5, 8)
        self.assertEqual(indices.shape, (1, 4))
        self.assertEqual(int(indices[0, 0]), 2)
        self.assertLess(float(distance[0, 0]), 1e-5)
        self.assertFalse(hasattr(bank, "actions"))

    def test_gaussian_costs_are_bounded_and_work_is_9002_blocks(self):
        import torch
        from anchored_planning.gaussian_cem import cem

        class Cost:
            def get_cost(self, info, actions):
                self.asserted = bool(torch.all(actions >= -1) and torch.all(actions <= 1))
                if not self.asserted:
                    raise AssertionError("CEM scored an illegal action")
                return (actions - .4).square().sum(dim=(-1, -2))

        class IdentityScaler:
            def transform(self, x):
                return x

            def inverse_transform(self, x):
                return x

        cost = Cost()
        native = SimpleNamespace(device=torch.device("cpu"), action_dim=2,
            pixels=lambda rgb: torch.zeros((1, 1, 3, 1, 1)), latent_cost=cost, scaler=IdentityScaler())
        bounds = np.array([[-1, -1], [1, 1]], np.float32)
        current = target = torch.zeros((1, 2))
        actions, detail = cem(native, None, current, target,
            torch.Generator().manual_seed(17), np.zeros((5, 2), np.float32), bounds, bounds)
        self.assertEqual((detail["predictor_batch_calls"], detail["predictor_candidate_transitions"]), (31, 9002))
        self.assertTrue(detail["refined_mean_selected"])
        self.assertLess(detail["selected_cost"], detail["prior_cost"])
        np.testing.assert_allclose(actions, .4, atol=.08)


if __name__ == "__main__":
    unittest.main()
