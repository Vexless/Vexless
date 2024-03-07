import numpy as np
import scipy.stats as stats
import pickle
from enum import Enum, auto
from typing import List, Optional, Tuple
import random

uint32_max = 2 ** 32 - 1
console_log = print

class mode_t(Enum):
    get_distribution = auto()
    generate = auto()
    
class distribution_t:   
    zipf = 0
    normal = 1
    uniform = 2
    poisson = 3
    inv_gaussian = 4
    
def bounded_inverted_gaussian(x, mean=0, std=1, height=1, domain=(-3, 3)):
    if x < domain[0] or x > domain[1]:
        return 0
    return height - np.exp(-((x - mean) ** 2) / (2 * std ** 2))

def normalize_bounded_function(domain=(-3, 3), height=1, mean=0, std=1):
    from scipy.integrate import quad
    area, _ = quad(bounded_inverted_gaussian, domain[0], domain[1], args=(mean, std, height, domain))
    return lambda x: bounded_inverted_gaussian(x, mean, std, height, domain) / area

def bounded_rejection_sampling(a, n):
    domain=(0, 1)
    x_samples = np.zeros(n)
    mean = random.random() * 1.6 - .3
    rvs = normalize_bounded_function(domain, 1, mean, a)
    i = 0
    while i < n:
        x_proposal = np.random.uniform(*domain)
        y_proposal = np.random.uniform(0, 1)
        if y_proposal < rvs(x_proposal):
            x_samples[i] = x_proposal
            i += 1
    return x_samples


def truncnorm_01rand(a, n):
    mu =  random.random() * 1.8 - .4
    lower = -mu/a
    upper = (1 - mu)/a
    return stats.truncnorm.rvs(lower, upper, loc=mu, scale=a, size=n)

def mix(v, b, n):
    return v
    # return (1 - b) * v + b * np.full(n, 1./n)

distribution_f = {
    distribution_t.zipf: lambda a, b, n: mix(np.random.zipf(a, n), b, n),
    distribution_t.normal: lambda a, b, n: mix(truncnorm_01rand(a, n), b, n),
    distribution_t.inv_gaussian: lambda a, b, n: bounded_rejection_sampling(a, n),
    distribution_t.uniform: lambda _, __, n: np.full(n, 1./n),
    distribution_t.poisson: lambda gamma, b, n: mix(np.random.poisson(gamma, n), b, n)
}

# distribution_f2 = {
#     distribution_t.zipf: lambda a, b, n: mix(np.random.zipf(a, n), b, n),
#     distribution_t.normal: lambda a, b, n: mix(norm2(a, n), b, n),
#     distribution_t.uniform: lambda _, __, n: np.full(n, 1./n),
#     distribution_t.poisson: lambda gamma, b, n: mix(np.random.poisson(gamma, n), b, n)
# }

distribution_name = {
    distribution_t.zipf: 'zipf',
    distribution_t.normal: 'normal',
    distribution_t.uniform: 'uniform',
    distribution_t.poisson: 'poisson',
    distribution_t.inv_gaussian: 'inverse gaussian'
}

class parameters_t:
    def __init__(self):
        self.outer = distribution_t.zipf
        self.inner = distribution_t.zipf
        self.a1 : float = 1.005
        self.a2 : float = 1.001
        self.b1 : float = 0
        self.b2 : float = 0
        self.b3 : float = 0
        self.b4 : float = 0.1
        self.s1 = 0
        self.n : int = 50000
        self.duration : int = 1000
        self.offset : float = 0
        self.granularity : int = 10
        self.shfl : float = 0
        self.mode : mode_t = mode_t.get_distribution
        self.f : str = './plan.bin'
        self.seeds : Tuple[int, int] = (0, 0)

class dump_t:
    def __init__(self, parameters : Optional[parameters_t] = None, plan : Optional[List[float]] = None):
        self.parameters : parameters_t = parameters
        self.plan : List[float] = plan
