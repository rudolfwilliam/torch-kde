"""Tests that check whether the kernel density estimator behaves as expected."""

from itertools import product
from functools import partial
import math
import random
import unittest

import torch
import numpy as np
from scipy.special import gamma
from torch.autograd import gradcheck

from torchkde.kernels import *
from torchkde.modules import KernelDensity
from torchkde.bandwidths import SUPPORTED_BANDWIDTHS

BANDWIDTHS = [1.0, 5.0] + SUPPORTED_BANDWIDTHS
DIMS = [1, 2]
TOLERANCE = 1e-1
WEIGHTS = [False, True]

DEVICES = ["cpu"]

N1 = 100
N2 = 10
GRID_N = 1000
GRID_RG = 100


class TestKDE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def test_integral(self):
        """Test that the kernel density estimator integrates to 1."""
        for kernel_str, bandwidth, dim, weights in product(SUPPORTED_KERNELS, BANDWIDTHS, DIMS, WEIGHTS):
            if kernel_str == 'von-mises-fisher':
                # Skip the von-mises-fisher kernel, must be handled differently
                # as it is not defined in the same way as the other kernels
                continue
            X = sample_from_gaussian(dim, N1)
            # Fit a kernel density estimator to the data
            kde = KernelDensity(bandwidth=bandwidth, kernel=kernel_str)
            if weights:
                weights = torch.rand((N1,)).exp()
                _ = kde.fit(X, sample_weight=weights)
            else:
                _ = kde.fit(X)
            # assess whether the kernel integrates to 1
            # evaluate the kernel density estimator at a grid of 2D points
            # Create ranges for each dimension
            ranges = [torch.linspace(-GRID_RG, GRID_RG, GRID_N) for _ in range(dim)]
            # Create the d-dimensional meshgrid
            meshgrid = torch.meshgrid(*ranges, indexing='ij')  # 'ij' indexing for Cartesian coordinates

            # Convert meshgrid to a single tensor of shape (n_points, d)
            grid_points = torch.stack(meshgrid, dim=-1).reshape(-1, dim)
            probs = kde.score_samples(grid_points).exp()
            delta = (GRID_RG * 2) / GRID_N
            integral = probs.sum() * (delta**dim)
            self.assertTrue((integral - 1.0).abs() < TOLERANCE, 
                            f"""Kernel {kernel_str}, for dimensionality {str(dim)} 
                            and bandwidth {str(bandwidth)} does not integrate to 1.""")
    
    def test_vmf_integral(self): # von-Mises-Fisher must be tested differently from other kernels
        for bandwidth, dim, weights in product(BANDWIDTHS, DIMS, WEIGHTS):
            if dim == 1 or type(bandwidth) == str:
                # Skip the von-mises-fisher kernel, must be handled differently
                # as it is not defined in the same way as the other kernels
                continue
            X = sample_from_gaussian(dim, N1)
            X = X / X.norm(dim=1, keepdim=True) # project onto sphere
            # Fit a kernel density estimator to the data
            kde = KernelDensity(bandwidth=bandwidth, kernel='von-mises-fisher')
            if weights:
                weights = torch.rand((N1,)).exp()
                _ = kde.fit(X, sample_weight=weights)
            else:
                _ = kde.fit(X)
            # assess whether the kernel integrates to 1

            # Create the d-dimensional meshgrid
            mesh_samples = sample_from_gaussian(dim, GRID_N**dim)
            mesh_samples = mesh_samples / mesh_samples.norm(dim=1, keepdim=True) # project onto sphere

            probs = kde.score_samples(mesh_samples).exp()
            surface_area = 2 * math.pi ** (dim / 2) / gamma(dim / 2)
            integral = probs.mean() * surface_area
            self.assertTrue((integral - 1.0).abs() < TOLERANCE, 
                            f"""Kernel von-mises-fisher, for dimensionality {str(dim)} 
                            and bandwidth {str(bandwidth)} does not integrate to 1.""")

                    
    def test_diffble(self, bandwidth=torch.tensor(1.0), eps=1e-07):
        """Test that the kernel density estimator is differentiable."""
        for kernel_str, dim in product(SUPPORTED_KERNELS, DIMS):
            def fit_and_eval(X, X_new, bandwidth):
                kde = KernelDensity(bandwidth=bandwidth, kernel=kernel_str)
                _ = kde.fit(X)
                return kde.score_samples(X_new)
            X = sample_from_gaussian(dim, N1).to(torch.float64) # relevant for the gradient check to convert to double
            X_new = sample_from_gaussian(dim, N2).to(torch.float64)

            if kernel_str == "von-mises-fisher": # normalization required
                if dim == 1:
                    continue
                # Project the data onto the unit sphere
                X = X / X.norm(dim=1, keepdim=True)
                X_new = X_new / X_new.norm(dim=1, keepdim=True)

            bandwidth = bandwidth.to(torch.float64)

            X.requires_grad = True
            X_new.requires_grad = False
            bandwidth.requires_grad = False

            # Check that the kernel density estimator is differentiable w.r.t. the training data
            fnc = partial(fit_and_eval, X_new=X_new, bandwidth=bandwidth)
            self.assertTrue(gradcheck(lambda X_: fnc(X=X_), (X,), raise_exception=False, eps=eps), 
                            f"""Kernel {kernel_str}, for dimensionality {str(dim)} is not differentiable w.r.t training data.""")
            
            if not kernel_str == "von-mises-fisher": # normalization required
                X.requires_grad = False
                X_new.requires_grad = False
                bandwidth.requires_grad = True

                # Check that the kernel density estimator is differentiable w.r.t. the bandwidth
                fnc = partial(fit_and_eval, X=X, X_new=X_new)
                self.assertTrue(gradcheck(lambda bandwidth_: fnc(bandwidth=bandwidth_), (bandwidth,), raise_exception=False, eps=eps), 
                                f"""Kernel {kernel_str}, for dimensionality {str(dim)} is not differentiable w.r.t. the bandwidth.""")
            
            X.requires_grad = False
            X_new.requires_grad = True
            bandwidth.requires_grad = False

            # Check that the kernel density estimator is differentiable w.r.t. the evaluation data
            fnc = partial(fit_and_eval, X=X, bandwidth=bandwidth)
            self.assertTrue(gradcheck(lambda X_new_: fnc(X_new=X_new_), (X_new,), raise_exception=False, eps=eps), 
                            f"""Kernel {kernel_str}, for dimensionality {str(dim)} is not differentiable w.r.t evaluation data.""")
            

def sample_from_gaussian(dim, N):
    # sample data from a normal distribution
    mean = torch.zeros(dim)
    covariance_matrix = torch.eye(dim) 

    # Create the multivariate Gaussian distribution
    multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    X = multivariate_normal.sample((N,))
    return X


if __name__ == "__main__":
    torch.manual_seed(0) # ensure reproducibility
    unittest.main()
