## Levenberg-Marquardt Alogorithm
from numpy import eye 
from numpy import zeros_like

from numpy.linalg import norm, inv



class LevenbergMarquardt:


    def __init__(self, lam, tol, itr):

        self.lam = lam
        self.tol = tol
        self.itr = itr


    def solve(self, func, jacb, x0):
        '''
        root = levenbergMarquardt( 
            func, jacb, x0, args
        )

        Finds a root of f(x) = 0
        func: f(x), vector variable -> vector valued
        jacb: jacobian matrix of f(x)
        x0  : initial condition
        args: arguments of func, jacb

        *** x is vector ***
        '''
        lam = self.lam
        tol = self.tol
        itr = self.itr

        _xK = x0
        uxK = zeros_like( x0 )
        _f = func( _xK )        ## object function
        pf = zeros_like( _f )   ## previous object

        ## initialize lambda
        dim = len( _f )
        lam = lam * eye( dim )

        for _ in range( itr ):
            Df = jacb( _xK )
            ## 1. check optimality condition
            opt_cond = Df.T @ _f

            if ( norm( opt_cond ) < tol ):
                return _xK

            ## 2. update xK
            dx = inv( Df.T @ Df + lam ) @ opt_cond

            uxK[:] = _xK - dx

            ## remember previous objects
            pf[:] = _f

            ## re evaluate objects
            _f = func( uxK )

            ## 3. check tentative iterate
            if ( norm( _f ) < norm( pf ) ):
                ## update xK
                _xK[:] = uxK
                lam *= 0.8
            else:
                ## do not update xK
                _f[:] = pf
                lam *= 2.0

        print('Levenberg-Marquardt fail to find root of given function')