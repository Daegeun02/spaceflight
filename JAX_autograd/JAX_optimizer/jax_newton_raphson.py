from jax import grad

from jax.numpy import abs



class NewtonRaphson:

    
    def __init__(self, tol, itr):

        self.tol = tol
        self.itr = itr


    def solve(self, _func, _grad, x0, force_return=False):
        
        tol = self.tol
        itr = self.itr

        xK = x0

        if ( _func( xK ) == 0.0 ):
            return xK

        for _ in range( itr ):
            f = _func( xK )
            g = _grad( xK )

            if ( abs( f ) < tol ):
                return xK
            
            if ( g == 0.0 ):
                print( 'zero division error' )
                break

            dx = f / g

            xK -= dx

        if force_return:
            return xK

        print( 'NewtonRaphson fail to find root of given function' )


    def _grad(self):

        _grad = grad( self.solve )

        return _grad