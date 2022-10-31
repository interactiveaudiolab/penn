import torch


###############################################################################
# Adaptive gradient clipping
###############################################################################


class AdaptiveGradientClipping(torch.optim.Optimizer):

    def __init__(
        self,
        parameters,
        optim: torch.optim.Optimizer,
        clipping: float = 1e-2,
        eps: float = 1e-3):
        self.optim = optim
        self.parameters = parameters
        self.eps = eps
        self.clipping = clipping
        self.param_groups = optim.param_groups
        self.state = optim.state

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step"""
        for p in self.parameters:
            for param in p:
                if param.grad is None:
                    continue

                # Get threshold
                param_norm = torch.max(
                    unitwise_norm(param.detach()),
                    torch.tensor(self.eps).to(param.device))
                max_norm = param_norm * self.clipping

                # Determine which gradients are above threshold
                grad_norm = unitwise_norm(param.grad.detach())
                trigger = grad_norm > max_norm

                # Clip gradients above threshold
                clipped_grad = param.grad * \
                    (max_norm / torch.max(
                        grad_norm,
                        torch.tensor(1e-6).to(grad_norm.device)))
                param.grad.detach().data.copy_(
                    torch.where(trigger, clipped_grad, param.grad))

        return self.optim.step()

    def zero_grad(self):
        for p in self.parameters:
            for param in p:
                if param.grad is not None:
                    if param.grad.grad_fn is not None:
                        param.grad.detach_()
                    else:
                        param.grad.requires_grad_(False)
                    param.grad.zero_()


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
