## Levenberg-Marquardt Algorithm
from numpy import eye 
from numpy import zeros_like

from numpy.linalg import norm, inv, cond, eig



class LevenbergMarquardt:


    def __init__(self, lam, tol, itr):

        self.lam = lam
        self.tol = tol
        self.itr = itr


    def solve(self, func, jacb, x0, lam=-1):
        '''
        root = levenbergMarquardt( 
            func, jacb, x0, args
        )

        Finds a root of f(x) = 0
        func: f(x), vector variable -> vector valued
        jacb: jacobian matrix of f(x)
        x0  : initial condition
        lam : lambda for regularizer

        *** x is vector ***
        '''
        if ( lam == -1 ):
            lam = self.lam
        else:
            lam = lam
        tol = self.tol
        itr = self.itr

        _xK = x0
        uxK = zeros_like( x0 )
        _f = func( _xK )        ## object function
        pf = zeros_like( _f )   ## previous object

        ## initialize lambda
        dim = len( _f )
        lam = lam * eye( dim )

        for _ in range( 500 ):
            Df = jacb( _xK )
            ## 1. check optimality condition
            opt_cond = Df.T @ _f

            # print( Df[2:4,:] )
            # print( Df.shape )

            if ( norm( opt_cond ) < tol ):
                return _xK
            # print( norm ( opt_cond ) )

            ## 2. update xK
            dx = inv( Df.T @ Df + lam ) @ opt_cond

            # print( cond( Df.T @ Df + lam ) )
            # print( eig( Df.T @ Df + lam )[0] )

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

        print( opt_cond )
        print('Levenberg-Marquardt fail to find root of given function')