'''
Contains common copulas: Clayton, Frank, Joe, Gumbel
Modified from ACNet, source code: https://github.com/lingchunkai/ACNet/blob/main/phi_listing.py
Recommend using HACopula toolbox instead, source code: https://github.com/gorecki/HACopula
'''

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class PhiListing(nn.Module):
    def __init__(self):
        super(PhiListing, self).__init__()

    def sample(self, ndims, n):
        # Marshall-Olkin 1988 ``Families of Multivariate Distributions''
        shape = (n, ndims)
        if hasattr(self, 'sample_M'):
            M = self.sample_M(n)[:, None].expand(-1, ndims)
            e = torch.distributions.exponential.Exponential(torch.ones(shape))
            E = e.sample()
            return self.forward(E/M)
        else:
            print("sample_M not yet implemented")
            return torch.ones(shape)*float('nan')

    
class ClaytonPhi(PhiListing):
    def __init__(self, theta):
        super(ClaytonPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = (1+t)**(-1/theta)
        return ret
    
    def inverse(self, u):
        theta = self.theta
        ret = u**(-theta)-1
        return ret
    
    def diff(self,t):
        theta = self.theta
        ret = (-1/theta)*(1+t)**(-1/theta-1.0)
        return ret

    def sample_M(self, n):
        m = torch.distributions.gamma.Gamma(1./self.theta, 1.0)
        return m.sample((n,))

    def pdf(self, X):
        """ 
        Differentiate CDF
        [From Wolfram]
        d/dx((d(x^(-z) + y^(-z) - 1)^(-1/z))/(dy)) = (-1/z - 1) z (-x^(-z - 1)) y^(-z - 1) (x^(-z) + y^(-z) - 1)^(-1/z - 2)
        """
        assert X.shape[1] == 2

        Z = X[:, 0]**(-self.theta) + X[:, 1]**(-self.theta) - 1.
        ret = torch.zeros_like(Z)
        ret[Z > 0] = (-1/self.theta-1.) * self.theta * -X[Z > 0, 0] ** (-self.theta-1) * X[Z > 0, 1] ** (
            -self.theta-1) * (X[Z > 0, 0] ** (-self.theta) + X[Z > 0, 1] ** (-self.theta) - 1) ** (-1./self.theta-2)

        return ret

    def cdf(self, X):
        assert X.shape[1] == 2

        return (torch.max(X[:, 0]**(-self.theta) + X[:, 1]**(-self.theta) - 1, torch.zeros(X.shape[0])))**(-1./self.theta)    
    
    
class FrankPhi(PhiListing):
    def __init__(self, theta):
        super(FrankPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = -1/theta * torch.log(torch.exp(-t)*(torch.exp(-theta)-1)+1)
        return ret
    
    def inverse(self, u):
        theta = self.theta
        ret = -torch.log((torch.exp(-theta*u)-1)/(torch.exp(-theta)-1))
        return ret

    def pdf(self, X):
        return None

    def cdf(self, X):
        return -1./self.theta * \
            torch.log(
                1 + (torch.exp(-self.theta * X[:, 0]) - 1) * (
                    torch.exp(-self.theta * X[:, 1]) - 1) / (torch.exp(-self.theta) - 1)
            )


class JoePhi(PhiListing):
    """
    The Joe Generator has a derivative that goes to infinity at t = 0. 
    Hence we need to be careful when t is close to 0!
    """

    def __init__(self, theta):
        super(JoePhi, self).__init__()

        self.eps = 0
        self.eps_snap_zero = 1e-15
        self.theta = nn.Parameter(theta)

    def forward(self, t):
        eps = self.eps
        if torch.any(t < eps):
            """
            logging.warning('''some entry in t is too small, < %s. May encounter numerical errors if taking gradients.
                            Smallest t= %s. Will be adding eps= %s to inputs for stability.''' % (eps, torch.min(t), eps))
            """
            t_ = t + eps
            
        theta = self.theta
        ret = 1-(1-torch.exp(-t))**(1/theta)
        return ret
    
    def inverse(self,t):
        return -torch.log(1-torch.pow(1-t,self.theta))

    def sample_M(self, n):
        U = torch.rand(n)
        ret = torch.ones_like(U)

        ginv_u = self.Ginv(U)
        cond = self.F(torch.floor(ginv_u))

        cut_indices = U <= (1./self.theta)
        z = cond < U
        j = cond >= U

        ret[z] = torch.ceil(ginv_u[z])
        ret[j] = torch.floor(ginv_u[j])
        ret[cut_indices] = 1.

        return ret

    def Ginv(self, y):
        return torch.exp(-self.theta * (torch.log(1.-y) + torch.lgamma(1.-1/self.theta)))

    def gamma(self, x):
        return torch.exp(torch.lgamma(x))

    def lbeta(self, x, y):
        return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x+y)

    def F(self, n):
        return 1. - 1. / (n * torch.exp(self.lbeta(n, 1.-1/self.theta)))

    def pdf(self, X):
        assert X.shape[1] == 2

        X_ = -X+1.0
        X_1 = X_[:, 0]
        X_2 = X_[:, 1]

        ret = -X_1 ** (self.theta-1) * X_2 ** (self.theta-1) * \
            ((X_1**self.theta) - (X_1**self.theta - 1) * X_2**self.theta)**(1./self.theta-2) * \
            ((X_1**self.theta-1) * (X_2**self.theta-1) - self.theta)

        return ret

    def cdf(self, X):
        assert X.shape[1] == 2

        X_ = -X+1.0
        X_1 = X_[:, 0]
        X_2 = X_[:, 1]

        return 1.0 - (X_1**self.theta + X_2**self.theta - (X_1**self.theta)*(X_2**self.theta))**(1./self.theta)


class GumbelPhi(PhiListing):
    def __init__(self, theta):
        super(GumbelPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = torch.exp(-((t) ** (1/theta)))
        return ret

    def pdf(self, X):
        assert X.shape[1] == 2

        u_ = (-torch.log(X[:, 0]))**(self.theta)
        v_ = (-torch.log(X[:, 1]))**(self.theta)

        return torch.exp(-(u_+v_)) ** (1/self.theta)


class IGPhi(nn.Module):
    def __init__(self, theta):
        super(IGPhi, self).__init__()

        self.theta = nn.Parameter(theta)

    def forward(self, t):
        theta = self.theta
        ret = torch.exp((1-torch.sqrt(1+2*theta**2*t))/theta)
        return ret
