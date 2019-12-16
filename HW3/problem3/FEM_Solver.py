"""
File: FEM_Solver.py

Author: Terrence Alsup
Date: June 24, 2019

Create an instance of a finite element solver for the elliptic pde.
Contains many useful helper functions as well.

This code can only handle the case in the report with Dirichlet boundary
condition on the left and Neumann on the right.

The functions contained within the class along with brief descriptions are
below:

__init__    :   Create an instance of the finite element solver.
basis_fun   :   Evaluate a basis function at a local node and element.
basis_deriv :   Evaluate the derivative of a basis function at a local node.
getAllNodes :   Get all nodes.
global_basis_fun    :   Evaluate a basis function over all elements.
global_basis_deriv  :   Evaluate the derivative of a basis function over all elements.
assemble    :   Assemble the matrix-vector system of equations.
convertLowerFormToMatrix    :   Convert the matrix A to a usable form.
constructLocalSystem    :   Construct the local system of equations.
solve   :   Solve the matrix system of equations.
interp_fun  :   Interpolate a given function with the basis functions.
interp_solution :   Interpolate the finite element solution.
interp_derivative   :   Interpolate the derivative of the finite element solution.
multiplyA   :   Multiply the stiffness matrix A by a vector U.
error : Compute the error in some norm w.r.t. a given function.
energyNorm : Compute the energy norm of a given function.
"""
import numpy as np
import EllipticPDE
from matplotlib import pyplot as plt

from scipy import linalg
from scipy import integrate
from scipy import sparse


class FEM_Solver:
    """
    Create an instance of a PDE solver using the FEM.

    pde is an EllipticPDE object

    N is the number of elements on the domain. N >= 1

    triangulation is a vector of points 0 = x_0 < x_1 < ... < x_N = 1
    len(triangulation) >= 2

    degree is the degree of the basis functions to use.
    """
    def __init__(self, pde, N = 10, triangulation = None, degree = 1):

        # The elliptic pde of the problem.
        self.pde = pde

        if triangulation is not None:
            self.triangulation = triangulation
            self.N = len(triangulation) - 1 # The number of elements.
        else:
            self.N = N
            self.triangulation = np.linspace(pde.a, pde.b, N + 1)

        # The degree of the Lagrange polynomials for interpolation.
        self.degree = degree

        # The number of basis functions.  Note that this is different than
        # the number of variables we solve for in the PDE since the Dirichlet
        # boundary condition implies that the coefficient on the first basis
        # function is zero.
        self.dof = self.N * self.degree + 1

        # The nodes corresponding to the degrees of freedom.
        self.nodes = self.getAllNodes()

        self.A = None # The stiffness matrix.
        self.F = None # The load vector.
        self.U = None # The coefficients of the finite element solution.


    """
    Evaluate the basis function i on element e at the points x.

    e = 0,...,N-1
    i = 0,...,d = degree
    x is a vector of points to evaluate at.
    """
    def basis_fun(self, e, i, x):

        # Get the triangulation of the domain.
        tri = self.triangulation
        p = self.degree
        N = self.N

        left_node = tri[e]
        right_node = tri[e + 1]

        nodes = np.linspace(left_node, right_node, p + 1)

        # Evaluate the Lagrange polynomial on the nodes.
        evals = (x >= left_node).astype(float) * (x <= right_node).astype(float)
        for j in range(p + 1):
            if j != i:
                evals *= (x - nodes[j]) / (nodes[i] - nodes[j])

        return evals

    """
    Evaluate the derivative of the basis function.

    e = 0,...,N-1
    i = 0,...,d = degree
    x is a vector of points to evaluate at.
    """
    def basis_deriv(self, e, i, x):
        # Get the triangulation of the domain.
        tri = self.triangulation
        p = self.degree
        N = self.N

        left_node = tri[e]
        right_node = tri[e + 1]

        local_nodes = np.linspace(left_node, right_node, p + 1)

        evals = np.zeros(x.shape)
        for j in range(p + 1):
            if j != i:
                prod = np.ones(x.shape)
                for m in range(p + 1):
                    if m != j and m != i:
                        prod *= (x - local_nodes[m])/(local_nodes[i] - local_nodes[m])
                evals += prod/(local_nodes[i] - local_nodes[j])
        evals *= (x >= left_node).astype(float) * (x <= right_node).astype(float)
        return evals


    """
    Assemble the matrix-vector system.

    n is the number of Gaussian quadrature points to use.
    """
    def assemble(self, n = 5):

        tri = self.triangulation # The triangulation.
        N = self.N      # The number of elements.
        p = self.degree # The degree of the Lagrange basis functions.

        dof = self.dof  # The number of degrees of freedom for the problem + 1.
        # Note that there is one less degree of freedom than the number of
        # basis functions because of the Dirichlet boundary condition on the
        # left of the interval.

        A = np.zeros((p + 1, dof)) # A will be in lower form for solveh_banded.
        F = np.zeros(dof)

        # Construct the global stiffness matrix and load vector by looping
        # over all of the elements.  Note that the first element is special
        # because of the Dirichlet boundary condition.

        for e in range(N):
            # Construct all of the local stiffness matrices and load vectors.
            A_local, F_local = self.constructLocalSystem(e, n = n)

            for j in range(p + 1):
                A[j, e*p:(e+1)*p + 1] += np.pad(np.diag(A_local, k = j), (0, j), 'constant')

            F[e*p:(e+1)*p+1] += F_local

        # Now trim the first row due to the Dirichlet boundary condition at 0.
        A = A[:,1:]
        F = F[1:]
        # Add the Neumann boundary condition on the right.
        F[-1] += (self.pde).r

        self.A = A
        self.F = F

        return [A, F]


    """
    Build the local stiffness matrix and load vector for the element e.

    e = 0,...,N - 1 is the element
    n is the number of Gaussian quadrature points to use
    """
    def constructLocalSystem(self, e, n = 5):

        tri = self.triangulation
        N = self.N
        p = self.degree

        f = (self.pde).f
        k = (self.pde).k

        A_local = np.zeros((p + 1, p + 1))
        F_local = np.zeros(p + 1)

        a = tri[e]
        b = tri[e + 1]

        for i in range(p + 1):
            temp = lambda x: self.basis_fun(e, i, x)
            F_local[i] = f.compute_integral(temp, a, b, n = n)
            for j in range(i, p + 1):
                temp = lambda x: self.basis_deriv(e, i, x) * self.basis_deriv(e, j, x)
                A_local[i,j] = k.compute_integral(temp, a, b, n = n)
                A_local[j, i] = A_local[i, j]

        return [A_local, F_local]

    """
    Get the array of all the nodes.
    """
    def getAllNodes(self):
        tri = self.triangulation
        N = self.N
        p = self.degree
        dof = self.dof

        nodes = np.zeros(dof)

        for e in range(N):
            nodes[e*p : (e+1)*p + 1] = np.linspace(tri[e], tri[e + 1], p + 1)

        return nodes

    """
    Evaluate the global basis function at node indx at the points x.

    indx = 0,...,Nd is the global node index
    x is the vector of points to evaluate at
    """
    def global_basis_fun(self, indx, x):
        tri = self.triangulation
        N = self.N
        p = self.degree
        nodes = self.nodes
        dof = self.dof

        evals = np.zeros(x.shape)

        if indx == 0:
            evals = self.basis_fun(0, 0, x)
        elif indx == dof - 1:
            evals = self.basis_fun(N - 1, p, x)
        else:
            if np.mod(indx, p) == 0:
                # We are on the exterior of an element.
                e = indx // p
                evals = self.basis_fun(e, 0, x) + self.basis_fun(e - 1, p, x)
            else:
                e = indx // p
                i = indx - e * p
                evals = self.basis_fun(e, i, x)

        return evals


    """
    Evaluate the global basis derivative at node indx at the points x.

    indx = 0,...,Nd is the global node index
    x is the vector of points to evaluate at
    """
    def global_basis_deriv(self, indx, x):
        tri = self.triangulation
        N = self.N
        p = self.degree
        nodes = self.nodes
        dof = self.dof

        evals = np.zeros(x.shape)

        if indx == 0:
            evals = self.basis_deriv(0, 0, x)
        elif indx == dof - 1:
            evals = self.basis_deriv(N - 1, p, x)
        else:
            if np.mod(indx, p) == 0:
                # We are on the exterior of an element.
                e = indx // p
                evals = self.basis_deriv(e, 0, x) + self.basis_deriv(e - 1, p, x)
            else:
                e = indx // p
                i = indx - e * p
                evals = self.basis_deriv(e, i, x)

        return evals



    """
    Interpolate a function f with the basis at the points x.

    f is a function to interpolate.
    x is the vector of points to evaluate at.
    """
    def interp_fun(self, f, x):

        tri = self.triangulation
        N = self.N
        p = self.degree

        nodes = self.getAllNodes()

        f_interp = np.zeros(len(x))
        f_coeff = f(nodes)

        for indx in range(len(nodes)):
            f_interp += f_coeff[indx] * self.global_basis_fun(indx, x)

        return f_interp

    """
    Interpolate the computed solution.

    U is the vector of coefficients.
    x is the vector of points to evaluate the interpolant at.
    """
    def interp_solution(self, U, x):

        tri = self.triangulation
        N = self.N
        p = self.degree
        dof = self.dof

        nodes = self.getAllNodes()

        u_interp = (self.pde).l * np.ones(len(x))

        for indx in range(1, dof):
            u_interp += U[indx - 1] * self.global_basis_fun(indx, x)

        return u_interp

    """
    Multiply the matrix A by the vector U.

    U is a vector of length Nd.

    Useful for computing the energy norm.
    """
    def multiplyA(self, U):
        A = self.convertLowerFormToMatrix()
        return A.dot(U)


    """
    Solve the system of equations AU = F and assemble the system if needed.
    """
    def solve(self):
        # Check if the system has already been assembled.
        A = self.A
        F = self.F
        # 1D case is handled separately due to solveh_banded limitations.
        if F.shape[0] == 1:
            U = np.asarray([F[0] / A[0,0]])
        else:
            U = linalg.solveh_banded(A, F, lower = True)
        self.U = U
        return U

    """
    Computes the p-norm of the error of the FEM solution.

    u_exact is the true solution (a function).

    Assume x_grid is equally spaced and a fine grid.
    0 = x_0 < x_1 < ... < x_N = 1

    p = 1, 2, np.inf, energy

    du_exact is the true derivative (a function).
    """
    def error(self, x_grid, u_exact, p = 2, du_exact = None):

        h = 1 / (x_grid[1] - x_grid[0])
        U = self.U

        err = lambda x: u_exact(x) - self.interp_solution(U, x)
        e = err(x_grid)

        if du_exact is not None and p == 'energy':
            # Compute the H1 energy norm of the difference.
            d_err = lambda x: du_exact(x) - self.interp_derivative(U, x)
            return self.energyNorm(d_err)

        if p == 1:
            return h * linalg.norm(e, 1)
        elif p == 2:
            return np.sqrt(h) * linalg.norm(e)
        else:
            # Infinity norm.
            return linalg.norm(e, np.inf)

    """
    Compute the H1 norm of a function with derivative du.

    du is the exact derivative.
    n is the number of Gaussian quadrature points to use.
    """
    def energyNorm(self, du, n = 500):
        k = (self.pde).k
        a = (self.pde).a
        b = (self.pde).b

        # Compute the H1 norm squared.
        diff2 = lambda x: du(x)**2
        norm2 = k.compute_integral(diff2, a, b, n = n)

        return np.sqrt(norm2)


    """
    Helper function to evaluate the derivative on a grid.

    U is a vector of length Nd storing the coefficients of the basis vectors.
    x is the grid of points we want to evaluate on.
    """
    def interp_derivative(self, U, x):

        tri = self.triangulation
        N = self.N
        p = self.degree
        dof = self.dof
        nodes = self.nodes

        if callable(U):
            U = U(nodes[1:])

        nodes = self.getAllNodes()

        u_interp = np.zeros(x.shape)

        for indx in range(1, dof):
            u_interp += U[indx - 1] * self.global_basis_deriv(indx, x)

        return u_interp
