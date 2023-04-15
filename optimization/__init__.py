## optimization algorithm's are implemented in this package
from .newton_raphson       import NewtonRaphson
from .levenberg_marquardt  import LevenbergMarquardt
from .augmented_lagrangian import AugmentedLagrangian



tol = 1.0e-8
itr = 500
newtonRaphson = NewtonRaphson( tol, itr )
newtonRaphson = newtonRaphson.solve


lam = 1
tol = 1.0e-7
itr = 500
levenbergMarquardt = LevenbergMarquardt( lam, tol, itr )
levenbergMarquardt = levenbergMarquardt.solve

lam = 1
tol = 1.0e-8
itr = 500
augmentedLagrangian = AugmentedLagrangian( lam, tol, itr, levenbergMarquardt )
augmentedLagrangian = augmentedLagrangian.solve