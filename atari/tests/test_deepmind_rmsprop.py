import unittest

import torch

from deepmind_rmsprop import DeepMindRMSprop


class DeepMindRMSpropTests(unittest.TestCase):
    def test_matches_manual_centered_no_momentum_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.5, -2.0], dtype=torch.float64))
        optimizer = DeepMindRMSprop([parameter], lr=0.25, alpha=0.95, eps=0.01)

        expected = parameter.detach().clone()
        grad_avg = torch.zeros_like(expected)
        grad_sq_avg = torch.zeros_like(expected)
        gradients = (
            torch.tensor([1.0, -2.0], dtype=torch.float64),
            torch.tensor([-0.5, 3.0], dtype=torch.float64),
        )

        for gradient in gradients:
            grad_avg = 0.95 * grad_avg + 0.05 * gradient
            grad_sq_avg = 0.95 * grad_sq_avg + 0.05 * gradient.square()
            expected -= 0.25 * gradient / torch.sqrt(grad_sq_avg - grad_avg.square() + 0.01)

            parameter.grad = gradient.clone()
            optimizer.step()

        torch.testing.assert_close(parameter.detach(), expected, rtol=0, atol=1e-12)

    def test_differs_from_pytorch_centered_rmsprop_at_epsilon_boundary(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float64))
        optimizer = DeepMindRMSprop([parameter], lr=1.0, alpha=0.95, eps=0.01)
        parameter.grad = torch.tensor([0.001], dtype=torch.float64)
        optimizer.step()

        # The released rule uses sqrt(variance + eps), not sqrt(variance) + eps.
        expected = torch.tensor(
            [-0.001 / torch.sqrt(torch.tensor(0.0000000475 + 0.01, dtype=torch.float64))],
            dtype=torch.float64,
        )
        torch.testing.assert_close(parameter.detach(), expected, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
