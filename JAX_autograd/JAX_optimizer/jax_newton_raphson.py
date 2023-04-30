from jax import grad

from jax.numpy import abs



class NewtonRaphson:

    
    def __init__(self, tol, itr):

        self.tol = tol
        self.itr = itr


    def solve(self, _func, x0):
        
        _grad = grad( _func )

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

        print( 'NewtonRaphson fail to find root of given function' )


    def _grad(self):

        _grad = grad( self.solve )

        return _grad