function [U,A,F] = FEM_solution(N)
% FEM_solution computes the finite element solution to the PDE
% -(u')' = sgn(x - 0.5) with 0 Dirichlet boundary conditions on (0,1).
%
% Assume that the x_i are evenly spaced with mesh width h, so that
% x_i = ih for i = 0,...,N.
%
% N is the number of elements.  Use piecewise quadratic basis functions.
%
% Returns the vectors U,F and the matrix A.

% Mesh width.
h = 1/N;

% Define the basis functions.  Note that we do not enfore the compact
% support here, but do it later during integration.

% Basis functions Phi_{2i-1} for i = 1,...,N
phi_2im1 = @(x, i) 4*(x-(i-1)*h).*(x - i*h)/(h^2);

% Basis functions Phi_{2i} for i = 1,...,N-1
phi_2i = @(x, i) 2*(x - (i-1)*h).*(x - (i - 0.5)*h)/(h^2) .* (x <= i*h) ...
     + 2*(x - (i+0.5)*h).*(x - (i+1)*h)/(h^2) .* (x > i*h);

% Derivative of basis functions Phi_{2i-1} for i = 1,...,N
dphi_2im1 = @(x, i) 4*((x-(i-1)*h) + (x - i*h))/(h^2);

% Derivative of basis functions Phi_{2i} for i = 1,...,N-1
dphi_2i = @(x, i) 2*((x - (i-1)*h) + (x - (i - 0.5)*h))/(h^2) .* (x <= i*h) ...
     + 2*((x - (i+0.5)*h) + (x - (i+1)*h))/(h^2) .* (x > i*h);

% Assembling the system.
A = sparse(2*N - 1, 2*N - 1);
F = zeros(2*N - 1, 1);

for i = 1:N
    A(2*i - 1,2*i - 1) = 
end
    

end

