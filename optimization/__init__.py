## optimization algorithm's are implemented in this package
from .newton_raphson import NewtonRaphson



tol = 1.0e-8
itr = 1000
newtonRaphson = NewtonRaphson( tol, itr )
newtonRaphson = newtonRaphson.solve