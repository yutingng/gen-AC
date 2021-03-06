"""
Modified from ACNet, source code: https://github.com/lingchunkai/ACNet/blob/main/phi_listing.py
Some functions have been optimized to work with gen-AC code and may no longer work with ACNet code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np

class PhiInv(nn.Module):
    def __init__(self, phi):
        super(PhiInv, self).__init__()
        self.phi = phi

    def forward(self, y, t0=None, max_iter=400, tol=1e-10):
        with torch.no_grad():
            t = newton_root(self.phi, y, max_iter=max_iter, tol=tol)

        topt = t.detach().clone().requires_grad_(True)
        # clone() = make a copy where gradients flow back to original 
        # clone().detach() = prevent gradient flow back to original
        # detach().clone() = computationally more efficient
        # requires_grad_(True) this new tensor requires gradient
        
        f_topt = self.phi(topt) # approximately equal to y
        return self.FastInverse.apply(y, topt, f_topt, self.phi)

    class FastInverse(torch.autograd.Function):
        '''
        forward: avoid running newton_root repeatedly.
        backward: specify gradients of PhiInv to PyTorch.

        In the backward pass, we provide gradients w.r.t 
        (i) `y`, and (ii) `w` via `f_topt=self.phi(topt)` approx equal to y,
        i.e., the function evaluated (with the current `w`) on topt. 
        Note that this should contain *values* approximately equal to y, 
        #but will have the necessary computational graph built up,
        #(beginning with topt.requires_grad_(True) and f_topt = self.phi(topt))
        #but detached from y, i.e. unrelated to y.
        
        https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
        '''
        @staticmethod
        def forward(ctx, y, topt, f_topt, phi):
            ctx.save_for_backward(y, topt, f_topt)
            ctx.phi = phi
            return topt

        @staticmethod
        def backward(ctx, grad):
            y, topt, f_topt = ctx.saved_tensors
            phi = ctx.phi

            with torch.enable_grad():
                # Call FakeInverse once again, 
                # to allow for higher order derivatives.
                z = PhiInv.FastInverse.apply(y, topt, f_topt, phi)

                # Find phi'(z), i.e., take derivatives of phi(z) w.r.t z.
                if hasattr(phi,'ndiff'):
                    dev_z = phi.ndiff(z,ndiff=1)
                else:
                    f = phi(z)
                    dev_z = torch.autograd.grad(f.sum(), z, create_graph=True)[0]

                # Refer to derivations for inverses.
                # Note when taking derivatives w.r.t. `w`, we make use of 
                # autograd's automatic application of the chain rule.
                # autograd finds the derivative d/dw[phi(z)], 
                # which when multiplied by the 3rd returned value,
                # gives the derivative d/dw[phi^{-1}].
                # Note that `w` is that contained by phi at f_topt.
                
                # what about gradients on "y" that is nested in phi_inverse?
                
                return grad/dev_z, None, -grad/dev_z, None  


def newton_root(phi, y, t0=None, max_iter=200, tol=1e-10):
    '''
    Solve
        f(t) = y
    using the Newton's root finding method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    t0: Tensor of shape `s` indicating the initial guess for the root.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the absolute difference |f(t) - y|.
        By default, this is set to 1e-14,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if t0 is None:
        t = torch.zeros_like(y) # why not 0.5?
    else:
        t = t0.detach().clone()

    s = y.size()
    for it in range(max_iter):


            
        if hasattr(phi,'ndiff'):
            f_t = phi(t)
            fp_t = phi.ndiff(t,ndiff=1)
        else:
            with torch.enable_grad():
                f_t = phi(t.requires_grad_(True))
                fp_t = torch.autograd.grad(f_t.sum(), t)[0]
        
        assert not torch.any(torch.isnan(fp_t))

        assert f_t.size() == s
        assert fp_t.size() == s

        g_t = f_t - y

        # Terminate algorithm when all errors are sufficiently small.
        if (torch.abs(g_t) < tol).all():
            break

        t = t - g_t / fp_t

    # error if termination criterion (tol) not met. 
    assert torch.abs(g_t).max() < tol, "t=%s, f(t)-y=%s, y=%s, iter=%s, max dev:%s" % (t, g_t, y, it, g_t.max())
    assert t.size() == s
    
    return t


def bisection_root(phi, y, lb=None, ub=None, increasing=True, max_iter=100, tol=1e-10):
    '''
    Solve
        f(t) = y
    using the bisection method.

    Parameters
    ----------
    f: Function which takes in a Tensor t of shape `s` and outputs
    the pointwise evaluation f(t).
    y: Tensor of shape `s`.
    lb, ub: lower and upper bounds for t.
    increasing: True if f is increasing, False if decreasing.
    max_iter: Positive integer containing the max. number of iterations.
    tol: Termination criterion for the difference in upper and lower bounds.
        By default, this is set to 1e-10,
        beyond which instability could occur when using pytorch `DoubleTensor`.

    Returns:
        Tensor `t*` of size `s` such that f(t*) ~= y
    '''
    if lb is None:
        lb = torch.zeros_like(y)
    if ub is None:
        ub = torch.ones_like(y)

    assert lb.size() == y.size()
    assert ub.size() == y.size()
    assert torch.all(lb < ub)

    f_ub = phi(ub)
    f_lb = phi(lb)
    assert torch.all(f_ub >= f_lb) or not increasing, 'Need f to be monotonically non-decreasing.'
    assert torch.all(f_lb >= f_ub) or increasing, 'Need f to be monotonically non-increasing.'

    assert (torch.all(
        f_ub >= y) and torch.all(f_lb <= y)) or not increasing, 'y must lie within lower and upper bound. max min y=%s, %s. ub, lb=%s %s' % (y.max(), y.min(), ub, lb)
    assert (torch.all(
        f_ub <= y) and torch.all(f_lb >= y)) or increasing, 'y must lie within lower and upper bound. y=%s, %s. ub, lb=%s %s' % (y.max(), y.min(), ub, lb)

    for it in range(max_iter):
        t = (lb + ub)/2
        f_t = phi(t)

        if increasing:
            too_low, too_high = f_t < y, f_t >= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]
        else:
            too_low, too_high = f_t > y, f_t <= y
            lb[too_low] = t[too_low]
            ub[too_high] = t[too_high]

        assert torch.all(ub - lb > 0.), "lb: %s, ub: %s" % (lb, ub)

    assert torch.all(ub - lb <= tol)
    return t


def bisection_default_increasing(phi, y):
    '''
    Wrapper for performing bisection method when f is increasing.
    '''
    return bisection_root(phi, y, increasing=True)


def bisection_default_decreasing(phi, y):
    '''
    Wrapper for performing bisection method when f is decreasing.
    '''
    return bisection_root(phi, y, increasing=False)

    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(20, 20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))

    
class Generator(nn.Module):
    ''' for vanilla GAN '''
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(20, 20),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        return torch.sigmoid(self.model(x))
    
    
class MixExpPhi(nn.Module):
    '''
    Sample net for phi involving the sum of N negative exponentials.
    phi(t) = m1 * exp(-w1 * t) + m2 * exp(-w2 * t) + ... + mN * exp(-wN * t)

    Network Parameters
    ==================
    mix: Tensor of size N such that such that (m1, m2, ..., mN) = softmax(mix)
    slope: Tensor of size N such that exp(m1) = w1, exp(m2) = w2, exp(mN) = wN

    Note that this implies
    i) m1, m2, ..., mN > 0 and m1 + m2 + ... + mN = 1.0
    ii) w1, w2, ..., wN > 0
    '''

    def __init__(self, init_w=None, N=2):
        super(MixExpPhiStatic, self).__init__()

        if init_w is None:
            self.mix = nn.Parameter(torch.log(torch.rand(N)), requires_grad=True)
            self.slope = nn.Parameter(torch.log(10)*(torch.rand(N)*6), requires_grad=True)
        else:
            assert len(init_w) == 2
            assert init_w[0].numel() == init_w[1].numel()
            self.mix = nn.Parameter(init_w[0])
            self.slope = nn.Parameter(init_w[1])

    def forward(self, t):
        s = t.size()
        t_ = t.flatten()
        nquery, nmix = t.numel(), self.mix.numel()

        mix_ = F.softmax(self.mix)
        exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                         torch.exp(self.slope)[None, :].expand(nquery, nmix))

        ret = torch.sum(mix_ * exps, dim=1)
        return ret.reshape(s)
    
    
class MixExpPhiStochastic(nn.Module):
    '''
    Sample net for phi involving the mean of Nz negative exponentials.
    phi(t) = mean(exp(-wi * t))

    Network Parameters
    ==================
    slope: Tensor of size Nz such that exp(m1) = w1, ..., exp(mN) = wN

    Note that this implies
    w1, ..., wN > 0
    '''

    def __init__(self, Nz=100):
        super(MixExpPhiStochastic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1)
        )
        self.M = self.sample_M(Nz)

    def resample_M(self, Nz):
        self.M = self.sample_M(Nz)
        
    def sample_M(self, N):
        return torch.exp(self.model(torch.rand(N).view(-1,1)).view(-1))
        
    def forward(self, t):
        s = t.size()
        t_ = t.flatten()
               
        nquery, nmix = t.numel(), self.M.numel()

        exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                         self.M[None, :].expand(nquery, nmix))

        ret = torch.mean(exps, dim=1)
        return ret.reshape(s)
    
    def ndiff(self, t, ndiff=0):
        s = t.size()
        t_ = t.flatten()
               
        nquery, nmix = t.numel(), self.M.numel()

        exps = torch.pow(-self.M[None, :].expand(nquery, nmix),ndiff) * \
                         torch.exp(-t_[:, None].expand(nquery, nmix) *
                         self.M[None, :].expand(nquery, nmix))

        ret = torch.mean(exps, dim=1)
        return ret.reshape(s)
    
    
class InnerGenerator(nn.Module):

    def __init__(self, OuterGenerator, Nz=100):
        super(InnerGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 10),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(10, 1)
        )
        self.mu = nn.Parameter(torch.rand(1))
        self.beta = nn.Parameter(torch.rand(1))
        self.M = self.sample_M(Nz)
        
        for param in OuterGenerator.parameters():
            param.requires_grad = False
        self.psi = OuterGenerator

    def resample_M(self, Nz):
        self.M = self.sample_M(Nz)
        
    def sample_M(self, N):
        return torch.exp(self.model(torch.rand(N).view(-1,1)).view(-1))
        
    def forward(self, t):
        s = t.size()
        t_ = t.flatten()
               
        nquery, nmix = t.numel(), self.M.numel()

        exps = torch.exp(-t_[:, None].expand(nquery, nmix) *
                         self.M[None, :].expand(nquery, nmix))

        ret = torch.exp(self.mu)*t_+torch.exp(self.beta)*(1-torch.mean(exps, dim=1))
        return self.psi(ret.reshape(s))


class Copula(nn.Module):
    def __init__(self, phi):
        super(Copula, self).__init__()
        self.phi = phi
        self.phi_inv = PhiInv(phi)

    def forward(self, y, mode='cdf', others=None, tol=1e-10):
        if not y.requires_grad:
            y.requires_grad_(True)
        ndims = y.size()[1]
        inverses = self.phi_inv(y, tol=tol)
        cdf = self.phi(inverses.sum(dim=1))

        if mode == 'cdf':
            return cdf

        if mode == 'pdf':
            cur = cdf
            for dim in range(ndims):
                # TODO: Take gradients with respect to only one dimension of y at at time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]
            return cur
        
        if mode == 'pdf2':
            numerator = self.phi.ndiff(inverses.sum(dim=1),ndiff=ndims)
            denominator = torch.prod(self.phi.ndiff(inverses,ndiff=1),dim=1)
            return numerator/denominator
        
        elif mode == 'cond_cdf':
            target_dims = others['cond_dims']

            # Numerator
            cur = cdf
            for dim in target_dims:
                # TODO: Take gradients with respect to one dimension of y at a time
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True, retain_graph=True)[0][:, dim]
            numerator = cur

            # Denominator
            trunc_cdf = self.phi(inverses[:, target_dims])
            cur = trunc_cdf
            for dim in range(len(target_dims)):
                cur = torch.autograd.grad(
                    cur.sum(), y, create_graph=True)[0][:, dim]

            denominator = cur
            return numerator/denominator
        
        
def sample(net, ndims, N, seed=142857):
    """
    Conditional sampling method: 
    (i)  compute conditional CDF, 
    (ii) compute inverse of conditional CDF to sample. 
    Very slow, only recommended for small dimensions, e.g. 2.
    Does *not* use efficient Marshall-Olkin 1988 method.
    """
    # Store old seed and set new seed
    old_rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    U = torch.rand(N, ndims)

    for dim in range(1, ndims):
        # don't have to sample dim 0
        print('Sampling from dim: %s' % dim)
        y = U[:, dim].detach().clone()

        def cond_cdf_func(u):
            U_ = U.detach().clone()
            U_[:, dim] = u
            cond_cdf = net(U_[:, :(dim+1)], "cond_cdf",
                           others={'cond_dims': list(range(dim))})
            return cond_cdf

        # Call inverse using the conditional cdf as the function.
        U[:, dim] = bisection_default_increasing(cond_cdf_func, y).detach()

    # Revert to old random state.
    torch.random.set_rng_state(old_rng_state)

    return U


def sampleStochastic(net, ndims, n):
    # Marshall-Olkin 1988 ``Families of Multivariate Distributions``
    
    M = net.phi.sample_M(n)[:, None].expand(-1, ndims)
    e = torch.distributions.exponential.Exponential(torch.ones((n, ndims)))
    E = e.sample()
    
    return net.phi.forward(E/M)


def sampleInner(phi, ndims, n, M0=None):
    if M0 is None:
        M0 = phi.psi.sample_M(n)
    
    njumps = torch.poisson(torch.exp(phi.beta)*M0).int()
    c = torch.tensor([phi.sample_M(njumps[i].item()).sum() for i in range(n)])
    lso = torch.exp(phi.mu)*M0+c
    
    M = lso[:,None].expand(-1,ndims)
    e = torch.distributions.exponential.Exponential(torch.ones((n, ndims)))
    E = e.sample()
    
    return phi.forward(E/M)
