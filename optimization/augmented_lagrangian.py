## Augmented Lagrangian Algorithm
from numpy import zeros



class AugmentedLagrangian:

    
    def __init__(self, lam, tol, itr, optimizer):

        self.lam = lam
        self.tol = tol
        self.itr = itr

        self.m = 1
        self.z = None

        ## generally use Levenberg Marquardt method
        self.optimizer = optimizer


    def solve(self, func, jacb, x0, n_obj, n_cns):
        '''
        minimizer = augmentedLagrangian(
            func, jacb, x0, n_obj, n_cns
        )

        Finds a minimizer of given problem

        minimize        f(x)^T f(x) + m * g(x)^T g(x) \\
        subject to      g(x) = 0

        Args:
            func (function): J(x)
                J(x) = f(x)^T f(x) + m * {g(x) + z/(2m)}^T {g(x) + z/(2m)}

            jacb (function): DJ(x)
                DJ(x) = dJ(x) / dx

            x0 (ndarray): initial value

            n_obj (int): dim of object function's output

            n_cns (int): dim of constraint functions' output
        '''

        lam = self.lam
        tol = self.tol
        itr = self.itr

        m = self.m
        z = zeros( n_cns )      ## lagrange multiplier

        optimizer = self.optimizer