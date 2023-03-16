## optimization algorithm's are implemented in this package
from .newton_raphson      import NewtonRaphson
from .levenberg_marquardt import LevenbergMarquardt



tol = 1.0e-8
itr = 1000
newtonRaphson = NewtonRaphson( tol, itr )
newtonRaphson = newtonRaphson.solve


lam = 1
tol = 1.0e-8
itr = 1000
levenbergMarquardt = LevenbergMarquardt( lam, tol, itr )
levenbergMarquardt = levenbergMarquardt.solve