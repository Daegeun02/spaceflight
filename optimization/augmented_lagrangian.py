## Augmented Lagrangian Algorithm
from math import sqrt 

from numpy import zeros, array

from numpy.linalg import norm



class AugmentedLagrangian:

    
    def __init__(self, lam, tol, itr, optimizer):

        self.lam = lam
        self.tol = tol
        self.itr = itr

        self.m = 1
        self.z = None

        ## generally use Levenberg Marquardt method
        self.optimizer = optimizer


    def solve(self, func, jacb, x0, n_obj, n_cns, lam=-1):
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
        if ( lam == -1 ):
            lam = self.lam
        else:
            lam = lam
        tol = self.tol
        itr = self.itr

        fK = zeros( n_obj + n_cns )
        gP = zeros( n_obj + n_cns )
        gK = zeros( n_obj + n_cns )
        DK = zeros((n_obj+n_cns,n_obj+n_cns))

        mK = array([self.m])
        zK = zeros( n_cns )      ## lagrange multiplier

        ALfunc = rebuild_func( func, zK, mK, n_obj, fK )
        ALjacb = rebuild_jacb( jacb, zK, mK, n_obj, DK )

        optimizer = self.optimizer

        xK = array( x0 ) 

        for _ in range( itr ):
            ## update x with levenberg marquardt algorithm
            xK[:] = optimizer( ALfunc, ALjacb, xK, lam=lam )

            fK[:] = ALfunc( xK[:], zK[:], mK )
            DK[:] = ALjacb( xK[:], zK[:], mK )
            ## optimality condition
            optimalCond = 2 * DK[:n_obj,:].T @ fK[:n_obj] + DK[n_obj:,:].T @ ( 2 * mK * fK[n_obj:] + zK )
            if ( norm( optimalCond ) < tol ):
                print( optimalCond )
                return xK

            ## update z 
            gK[:] = func( xK[:] )
            zK[:] = zK + 2 * mK * gK[n_obj:]

            ## update m
            if ( norm( gK[n_obj:] ) < ( 0.25 * norm( gP[n_obj:] ) ) ):
                pass
            else:
                mK *= 2

            gP[:] = gK[:]

        print( "Augmented Lagrangian couldn't find solution...")


def rebuild_func( func, z, m, n_obj, out ):

    def _func( x ):

        _m = sqrt( m )

        out[:] = func( x )

        out[n_obj:] = _m * out[n_obj:] + ( z / ( 2 * _m ) )

        return out

    return _func


def rebuild_jacb( jacb, z, m, n_obj, out ):

    def _jacb( x ):

        _m = sqrt( m )

        out[:] = jacb( x )

        out[n_obj:,:] = _m * out[n_obj:,:]

        return out

    return _jacb