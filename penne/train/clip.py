import torch


###############################################################################
# Adaptive gradient clipping
###############################################################################


class AdaptiveGradientClipping(torch.optim.Optimizer):
    """Adaptive Gradient Clipping"""

    def __init__(
        self,
        params,
        optim: torch.optim.Optimizer,
        clipping: float = 1e-2,
        eps: float = 1e-3):
        self.optim = optim
        self.clipping_parameters = params
        self.eps = eps
        self.clipping = clipping
        self.param_groups = optim.param_groups
        self.state = optim.state

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step"""
        for p in self.clipping_parameters:
            if p.grad is None:
                print('None gradient')
                continue
            param_norm = torch.max(
                unitwise_norm(p.detach()),
                torch.tensor(self.eps).to(p.device))
            grad_norm = unitwise_norm(p.grad.detach())
            max_norm = param_norm * self.clipping
            trigger = grad_norm > max_norm
            clipped_grad = p.grad * (max_norm / torch.max(
                grad_norm,
                torch.tensor(1e-6).to(grad_norm.device)))
            p.grad.detach().data.copy_(
                torch.where(trigger, clipped_grad, p.grad))
        return self.optim.step()


###############################################################################
# Utilities
###############################################################################


def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x ** 2, dim=dim, keepdim=keepdim) ** 0.5
