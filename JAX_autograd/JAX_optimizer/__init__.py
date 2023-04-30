from .jax_newton_raphson import NewtonRaphson



tol = 1.0e-8
itr = 500
newtonRaphson = NewtonRaphson( tol, itr )
newtonRaphson = newtonRaphson.solve