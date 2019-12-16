"""
Author: Terrence Alsup
Date: December 16, 2019

Solve the Bratu problem using Newton's method where the discretization is
linear finite elements on (0,1) with zero Dirichlet boundary data.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import fixed_quad
from scipy.linalg import solve

# Global variables for the number of elements.
N = 50
h = 1/N # mesh width
nodes = np.linspace(0, 1, N + 1)

# The parameter lambda in the Bratu equation.
lmda = 2

"""
Evaluate the basis function i at the points x.

x is a numpy array of shape (n, ) where n is the number of points
i is an integer between 0 and N with N being the number of elements
"""
def basis_fun(i, x):
    evals = (x-(i-1)*h)/h*(x >= (i-1)*h).astype(float)*(x <= i*h).astype(float)
    evals += ((i+1)*h-x)/h*(x <= (i+1)*h).astype(float)*(x > i*h).astype(float)
    return evals

"""
Evaluate the derivative of the basis function.

i = 0,...,N
x is a vector of points to evaluate at.
"""
def basis_deriv(i, x):
    if i == 0:
        evals = -1/h*(x <= h).astype(float)*(x >= 0).astype(float)
    elif i == N:
        evals = 1/h*(x >= (N-1)*h).astype(float)*(x <= 1).astype(float)
    else:
        evals = -1/h*(x <= (i+1)*h).astype(float)*(x > i*h).astype(float)
        evals += 1/h*(x >= (i-1)*h).astype(float)*(x <= i*h).astype(float)
    return evals

"""
Interpolate a function f with the basis at the points x.

f is a function to interpolate.
x is the vector of points to evaluate at.
"""
def interp_fun(f, x):
    f_interp = np.zeros(len(x))
    f_coeff = f(nodes) # Evaluate f at the nodes.
    for i in range(N + 1):
        f_interp += f_coeff[i] * basis_fun(i, x)
    return f_interp

"""
Interpolate the function with the coefficients U.

U is the vector of coefficients for 0 to N.
x is the vector of points to evaluate the interpolant at.
"""
def interp_coeff(U, x):
    u_interp = np.zeros(len(x))
    for i in range(N + 1):
        u_interp += U[i] * basis_fun(i, x)
    return u_interp

"""
Interpolate the derivative of the function with the coefficients U.

U is the vector of coefficients for 0 to N.
x is the vector of points to evaluate the interpolant at.
"""
def interp_coeff_deriv(U, x):
    du_interp = np.zeros(len(x))
    for i in range(N + 1):
        du_interp += U[i] * basis_deriv(i, x)
    return du_interp


"""
Build the local stiffness matrix and load vector for the element i.

i = 1,...,N is the element
"""
def assemble_local_stiff(i, u_k):

    A = np.zeros((2, 2))
    F = np.zeros(2)

    a = lambda e,j,x: -basis_deriv(e,x)*basis_deriv(j,x) + \
        lmda*np.exp(interp_coeff(u_k, x))*basis_fun(e,x)*basis_fun(j,x)

    A[0,0] = fixed_quad(lambda x: a(i-1, i-1, x), nodes[i-1], nodes[i])[0]
    A[1,0] = fixed_quad(lambda x: a(i, i-1, x), nodes[i-1], nodes[i])[0]
    A[0,1] = A[1,0] # Symmetric
    A[1,1] = fixed_quad(lambda x: a(i, i, x), nodes[i-1], nodes[i])[0]

    f = lambda e,x: interp_coeff_deriv(u_k, x)*basis_deriv(e, x) - \
        lmda*np.exp(interp_coeff(u_k, x))*basis_fun(e, x)

    F[0] = fixed_quad(lambda x: f(i-1, x), nodes[i-1], nodes[i])[0]
    F[1] = fixed_quad(lambda x: f(i, x), nodes[i-1], nodes[i])[0]

    return [A, F]

"""
Assemble the matrix-vector system.

n is the number of Gaussian quadrature points to use.
"""
def assemble(u_k):

    A = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    # Construct the global stiffness matrix and load vector by looping
    # over all of the elements.  Note that the first element is special
    # because of the Dirichlet boundary condition.

    for i in range(1, N + 1):
        # Construct all of the local stiffness matrices and load vectors.
        A_loc, F_loc = assemble_local_stiff(i, u_k)
        A[i-1:i+1,i-1:i+1] += A_loc
        F[i-1:i+1] += F_loc

    # Now trim the first row due to the Dirichlet boundary condition at 0.
    A = A[1:N,1:N]
    F = F[1:N]

    return [A, F]




"""
Compute the Newton update step.
"""
def newton_update(u_k):
    [A, F] = assemble(u_k)
    du = np.insert(solve(A, F), [0,N-1], 0)
    return du

def solve_bratu(u_init, mu = 1, iters = 10):
    u_k = u_init(nodes)
    for i in range(iters):
        du = newton_update(u_k)
        u_k += mu * du
    return u_k


"""
--------------------------------------------------------------------------------
Script to solve the problem starts here.
--------------------------------------------------------------------------------
"""
x = np.linspace(0,1,1000)


u_init1 = lambda x: np.zeros(x.shape)
u_k1 = solve_bratu(u_init1, mu = 1, iters = 10)

u_init2 = lambda x: 14*x*(1-x)
u_k2 = solve_bratu(u_init2, mu = 1, iters = 10)

plt.figure()
plt.plot(x, interp_coeff(u_k1, x), 'b', lw = 2, label = r'$u_{\mathrm{init}} = 0$')
plt.plot(x, interp_coeff(u_k2, x), 'r', lw = 2, label = r'$u_{\mathrm{init}} = 14x(1-x)$')
plt.xlabel(r'$x$')
plt.ylabel(r'$u$')
plt.title('Two solutions to Bratu\'s problem')
plt.legend()
plt.show()
