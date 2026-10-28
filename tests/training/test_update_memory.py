"""Sequential backward must preserve the previous summed-loss update."""
import copy
import unittest
from unittest.mock import patch

import torch

from test_dreamwaq_ppo import make_policy, make_alg, rollout


class TestSequentialBackward(unittest.TestCase):
    def test_matches_combined_backward_with_and_without_cenet_ppo_gradients(self):
        for connected in (False, True):
            with self.subTest(ppo_grad_to_cenet=connected):
                sequential = make_alg(make_policy(), ppo_grad_to_cenet=connected)
                sequential.init_storage('rl', 4, 6, [45], [8], [12])
                rollout(sequential)
                combined = copy.deepcopy(sequential)
                original_backward = torch.Tensor.backward
                pending = []

                def combined_backward(loss, *args, **kwargs):
                    pending.append(loss)
                    if len(pending) == 2:
                        total = pending[0] + pending[1]
                        pending.clear()
                        original_backward(total, *args, **kwargs)

                torch.manual_seed(71)
                sequential.update()
                torch.manual_seed(71)
                with patch.object(torch.Tensor, 'backward', combined_backward):
                    combined.update()
                self.assertFalse(pending)
                for a, b in zip(sequential.policy.parameters(), combined.policy.parameters()):
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
