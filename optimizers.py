import torch
from torch.optim import Optimizer


def _newton_schulz(G, steps=5):
    """Orthogonal polar factor of G via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-schulz.

    For 2D weight matrices: applies Nesterov momentum then orthogonalizes
    the update via Newton-Schulz, making the effective step a rotation rather
    than a raw gradient direction. Update is scaled by sqrt(max(n, m)) so the
    RMS step magnitude stays proportional to lr.

    For 1D / scalar params (biases): plain SGD with Nesterov momentum.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr  = group['lr']
            mu  = group['momentum']
            ns  = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)

                buf = state['buf']
                buf.mul_(mu).add_(g)
                update = g.add(buf, alpha=mu)   # Nesterov

                if update.ndim == 2:
                    update = _newton_schulz(update, steps=ns)
                    update = update * (max(update.size(0), update.size(1)) ** 0.5)

                p.add_(update, alpha=-lr)

        return loss


class ClampedCenteredAdam(Optimizer):
    """
    Adam variant that tracks true centered variance and clamps the denominator.

    Differences from standard Adam:
      - Second moment tracks E[(g - m)²] (true variance) instead of E[g²]
      - Denominator uses max(sqrt(variance), tau) instead of sqrt(v) + eps
        → acts as a hard floor on uncertainty; confident directions get larger steps
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), tau=1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 < tau:
            raise ValueError(f"Invalid tau: {tau}")
        defaults = dict(lr=lr, betas=betas, tau=tau)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            tau = group['tau']
            lr  = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ClampedCenteredAdam does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg']              = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_centered_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_centered_var = state['exp_avg'], state['exp_avg_centered_var']
                state['step'] += 1
                step = state['step']

                # First moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Centered variance: track (g - m)² instead of g²
                centered_grad = grad - exp_avg
                exp_avg_centered_var.mul_(beta2).add_(centered_grad.pow(2), alpha=1 - beta2)

                # Bias correction
                bc1 = 1 - beta1 ** step
                bc2 = 1 - beta2 ** step
                mean = exp_avg / bc1
                var  = exp_avg_centered_var / bc2

                # Hard floor on denominator: max(sqrt(var), tau)
                denom = var.sqrt().clamp_(min=tau)

                p.addcdiv_(mean, denom, value=-lr)

        return loss
