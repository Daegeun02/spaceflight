from .jax_newton_raphson import NewtonRaphson

from .jax_levenberg_marquardt import LevenbergMarquardt

from .gradient_descent import optimize



tol = 1.0e-8
itr = 500
newtonRaphson = NewtonRaphson( tol, itr )
newtonRaphson = newtonRaphson.solve


lam = 1
tol = 1.0e-7
itr = 500
levenbergMarquardt = LevenbergMarquardt( lam, tol, itr )
levenbergMarquardt = levenbergMarquardt.solve