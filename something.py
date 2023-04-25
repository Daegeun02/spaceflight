from optimization import levenbergMarquardt
from numpy import zeros
from numpy import exp



f = zeros( 2 )
J = zeros((2, 2))


def func( t ):
    
    def _func( X ):
        x, y = X
        
        f[0] = exp( y+t ) - x
        f[1] = exp( x+t ) + y
        
        return f

    return _func


def jacb( t ):

    def _jacb( X ):
        x, y = X

        J[0,0] = -1
        J[0,1] = exp( y+t )
        J[1,0] = exp( x+t )
        J[1,1] = 1

        return J
    
    return _jacb

_func = func( 1 )
_jacb = jacb( 1 )


x0 = zeros( 2 )

xS = levenbergMarquardt( _func, _jacb, x0 )

print( xS )