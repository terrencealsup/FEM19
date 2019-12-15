from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

mesh = IntervalMesh(3, 0, 1)
V = FunctionSpace(mesh, "Hermite", 3)

u = TrialFunction(V)
v = TestFunction(V)

a = dot(grad(u), grad(v))*dx + dot(u,v)*dx
A = assemble(a)
print(A)
