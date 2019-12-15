"""
Author: Terrence Alsup
Date: December 16, 2019

Solve the Bratu problem using Newton's method where the discretization is
linear finite elements on (0,1) with zero Dirichlet boundary data.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import fixed_quad

"""
Evaluate the basis function i at the points x.

x is a numpy array of shape (n, ) where n is the number of points
i is an integer between 1 and N - 1 with N being the number of elements
"""
def basis_fun(i, x):
    return

"""
Perform a single Newton iteration.  mu < 1 is a damping factor.
"""
def newton_iteration(u_k, mu = 1):
    return u_kp1
