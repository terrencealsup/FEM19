"""
FEM_solution.py

Author: Terrence Alsup
Date: Oct 4, 2019

Compute the finite element solution to the PDE
-u'' = sgn(x - 0.5) on (0,1) with u(0) = u(1) = 0.

We assume that the points x_i are evenly spaced with mesh width h = 1/N and
x_i = i*h.
"""
import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt


"""
Define the basis functions phi_k for k = 1,...,2N-1
"""
def phi(x, k, N):
    h = 1/N
    if np.mod(k,2) == 0:
        # Even basis function k = 2i.
        i = k//2
        y = 2*(x - (i-0.5)*h)*(x - (i-1)*h)/(h**2) * (x <= i*h).astype(float)
        y += 2*(x - (i+0.5)*h)*(x - (i+1)*h)/(h**2) * (x > i*h).astype(float)
        y *= (x > (i-1)*h).astype(float) * (x < (i+1)*h).astype(float)
        return y
    else:
        # Odd basis function k = 2i - 1.
        i = (k + 1)//2
        y = -4*(x - (i-1)*h)*(x - i*h)/(h**2)
        y *= (x > (i-1)*h).astype(float) * (x < i*h).astype(float)
        return y

"""
Define the derivatives of the basis functions.
"""
def dphi(x, k, N):
    h = 1/N
    if np.mod(k,2) == 0:
        # Even basis function k = 2i.
        i = k//2
        y = 2*((x - (i-0.5)*h) + (x - (i-1)*h))/(h**2) * (x <= i*h).astype(float)
        y += 2*((x - (i+0.5)*h)+(x - (i+1)*h))/(h**2) * (x > i*h).astype(float)
        y *= (x >= (i-1)*h).astype(float) * (x <= (i+1)*h).astype(float)
        return y
    else:
        # Odd basis function k = 2i - 1.
        i = (k + 1)//2
        y = -4*((x - (i-1)*h)+(x - i*h))/(h**2)
        y *= (x >= (i-1)*h).astype(float) * (x <= i*h).astype(float)
        return y

"""
Assemble the matrices A and F.

A is stored in a special way since it is a banded and Hermitian matrix.  See the
scipy.linalg.solveh_banded documentation for more details.
A will be stored in lower form.
"""
def assemble(N):

    h = 1/N # Mesh width.

    # Assemble the stiffness matrix A.
    A = np.zeros((3, 2*N - 1))

    # Get the diagonal entries.
    for k in range(1, 2*N): # 1,..,2N-1
        # Define the integrand.
        intgr = lambda x: dphi(x,k,N)**2
        # Get the limits of integration.
        if np.mod(k,2) == 0:
            i = k//2
            a = (i-1)*h
            b = (i+1)*h
            A[0,k-1] = integrate.fixed_quad(intgr, a, i*h, n = 5)[0]
            A[0,k-1] += integrate.fixed_quad(intgr, i*h, b, n = 5)[0]
        else:
            i = (k+1)//2
            a = (i-1)*h
            b = i*h
            # Integrate using 5 quadrature points.
            A[0,k-1] = integrate.fixed_quad(intgr, a, b, n = 5)[0]

    # Get the entries 1 above/below the diagonal.
    for k in range(1, 2*N - 1): # 1,...,2N-2
        # Define the integrand.
        intgr = lambda x: dphi(x,k,N)*dphi(x,k+1,N)
        # Get the limits of integration.
        if np.mod(k,2) == 0:
            i = k//2
            a = i*h
            b = (i+1)*h
        else:
            i = (k+1)//2
            a = (i-1)*h
            b = i*h
        # Integrate using 2 quadrature points.
        A[1,k-1] = integrate.fixed_quad(intgr, a, b, n = 5)[0]

    # Get the entries 2 above/below the diagonal.
    for k in range(1, 2*N - 2): # 1,...,2N-3
        # Only loop over evens since Phi_{2i+1} and Phi_{2i-1} don't overlap.
        if np.mod(k,2) == 0:
            i = k//2
            a = i*h
            b = (i+1)*h
            intgr = lambda x: dphi(x,k,N)*dphi(x,k+2,N)
            # Integrate using 2 quadrature points.
            A[2,k-1] = integrate.fixed_quad(intgr, a, b, n = 5)[0]

    # Assemble the vector F.
    F = np.zeros(2*N-1)
    for k in range(1, 2*N):
        intgr = lambda x: phi(x, k, N) * np.sign(x - 0.5)
        if np.mod(k,2) == 0:
            i = k//2
            if 0.5 > (i-1)*h and 0.5 < i*h:
                F[k-1] = integrate.fixed_quad(intgr, (i-1)*h, 0.5, n = 5)[0]
                F[k-1] += integrate.fixed_quad(intgr, 0.5, i*h, n = 5)[0]
            else:
                F[k-1] = integrate.fixed_quad(intgr, (i-1)*h, i*h, n = 5)[0]
            if 0.5 > i*h and 0.5 < (i+1)*h:
                F[k-1] += integrate.fixed_quad(intgr, i*h, 0.5, n = 5)[0]
                F[k-1] += integrate.fixed_quad(intgr, 0.5, (i+1)*h, n = 5)[0]
            else:
                F[k-1] = integrate.fixed_quad(intgr, i*h, (i+1)*h, n = 5)[0]
        else:
            i = (k+1)//2
            if 0.5 > (i-1)*h and 0.5 < i*h:
                F[k-1] = integrate.fixed_quad(intgr, (i-1)*h, 0.5, n = 5)[0]
                F[k-1] += integrate.fixed_quad(intgr, 0.5, i*h, n = 5)[0]
            else:
                F[k-1] = integrate.fixed_quad(intgr, (i-1)*h, i*h, n = 5)[0]
    return [A,F]

"""
Assemble and solve the system to get the vector U.
"""
def solve(N):
    [A,F] = assemble(N)
    U = linalg.solveh_banded(A, F, lower = True)
    return U


"""
Interpolate the finite element solution with N elements onto the fine grid x.
"""
def interpolate(x, N):
    u = np.zeros(x.shape)
    U = solve(N)
    for k in range(2*N-1):
        u += U[k]*phi(x,k+1,N)
    return u


"""
-----------------------------------------------------------------------------
Now plot the results for various N.
-----------------------------------------------------------------------------
"""


x = np.linspace(0,1,1000)
U_2 = interpolate(x, 2)
U_4 = interpolate(x, 4)
U_8 = interpolate(x, 8)
U_16 = interpolate(x, 16)



plt.figure(1)
plt.plot(x, U_2, 'g-', label = 'N = 2')
plt.plot(x, U_4, 'b-', label = 'N = 4')
plt.plot(x, U_8, 'r-', label = 'N = 8')
plt.plot(x, U_16, 'm-', label = 'N = 16')

plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('Finite element solutions for different N')
plt.legend()
plt.show()
