import torch
import torch.nn as nn
import math
from torch.optim.optimizer import Optimizer, required
import warnings


def signpower(x, q):
    return torch.abs(x) ** q * torch.sign(x)


def signpower_inplace(x, q):
    abs_x = torch.abs(x) ** q
    x.sign_()
    x.mul_(abs_x)


class Potential:
    def __init__(self): pass

    def __call__(self, w): pass

    def grad(self, w): pass

    def inv_grad(self, y, *args, **kwargs): pass

    def hessian(self, w): pass

    def bregman(self, w1, w2):
        return self(w1) - self(w2) - (self.grad(w2) * (w1 - w2)).sum()


class Pnorm(Potential):
    def __init__(self, p):
        super().__init__()
        self.p = p
        self.name = f'{p}-norm'

    def __call__(self, w):
        return (torch.abs(w) ** self.p).sum() / self.p

    def grad(self, w):
        return signpower(w, self.p - 1)

    def inv_grad(self, y, *args, **kwargs):
        return signpower(y, 1.0 / (self.p - 1))

    def hessian(self, w):
        return (self.p - 1) * torch.diag(torch.abs(w) ** (self.p - 2))

    def update_inplace(self, w, g, alpha):
        signpower_inplace(w, self.p - 1)
        w.add_(g, alpha=alpha)
        signpower_inplace(w, 1.0 / (self.p - 1))


class NegativeEntropy(Potential):
    def __init__(self):
        super().__init__()
        self.name = 'negative_entropy'

    def __call__(self, w):
        w = torch.abs(w)
        return (torch.log(w) * w).sum()

    def grad(self, w):
        return (torch.log(torch.abs(w)) + 1) * torch.sign(w)

    def inv_grad(self, y, sign_w, *args, **kwargs):
        return sign_w * torch.exp(y * sign_w - 1)

    def hessian(self, w):
        return torch.diag(1 / torch.abs(w))


class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, potential=None):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if potential is not None:
            if potential == 'negative_entropy':
                potential = NegativeEntropy()
            elif 'norm' in potential:
                potential = Pnorm(float(potential.split('-norm')[0]))
            else:
                raise NotImplementedError('Potential {} is not implemented'.format(potential))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, potential=potential)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            potential = group['potential']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,
                  d_p_list,
                  momentum_buffer_list,
                  weight_decay=weight_decay,
                  momentum=momentum,
                  lr=lr,
                  dampening=dampening,
                  nesterov=nesterov,
                  potential=potential)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss


def sgd(params, d_p_list, momentum_buffer_list, *, weight_decay, momentum, lr, dampening, nesterov, potential):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        if potential is None or (potential.name == 'p-norm' and potential.p == 2):
            param.add_(d_p, alpha=-lr)
        elif potential.name == 'negative_entropy':
            param.mul_(torch.exp(param.sign() * d_p * -lr))
        else:
            potential.update_inplace(param, d_p, -lr)
