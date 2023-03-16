## Levenberg-Marquardt Alogorithm
from numpy        import eye 
from numpy.linalg import norm, inv



class LevenbergMarquardt:


    def __init__(self, lam, tol, itr):

        self.lam = lam
        self.tol = tol
        self.itr = itr


    def solve(self, func, jacb, x0, args):
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

        val = func( x0, args )
        dim = len( val )

        lam = lam * eye( dim )

        if ( norm( val ) == 0.0 ):
            return x0

        xK = x0
        dx = 1e8

        for _ in range( itr ):
            f = func( xK, args )
            J = jacb( xK, args )

            if ( norm( f ) < tol ):
                return xK
            
            dx = inv( J.T @ J + lam ) @ J.T @ f

            xK -= dx

        print('Levenberg-Marquardt fail to find root of given function')