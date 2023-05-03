from geometry import MU

from optimization import newtonRaphson
from optimization import levenbergMarquardt

from numpy import array, zeros
from numpy import dot
from numpy import cos, sin, pi
from numpy import sqrt

from numpy.linalg import norm



def LM_func( _r0, _r1, _r2, out ):

    def _func( x ):

        a, b = x

        out[0] = _r1 * cos( a ) + _r2 * cos( b ) - _r0
        out[1] = _r1 * sin( a ) - _r2 * sin( b )

        return out
    
    return _func


def LM_jacb( _r0, _r1, _r2, out ):

    def _jacb( x ):

        a, b = x

        out[0,0] = _r1 * sin( a ) * ( -1 )
        out[1,0] = _r1 * cos( a )
        out[0,1] = _r2 * sin( b ) * ( -1 )
        out[1,1] = _r2 * cos( b ) * ( -1 )

        return out

    return _jacb

def NL_func( r0, v0, mu, a, t ):

    sqrt_m = sqrt( mu )
    sqrt_a = sqrt( a )

    _r0 = norm( r0 )

    def _func( x ):

        out = 0.0
        out += a * ( x - sqrt_a * sin( x / sqrt_a ) )
        out += ( dot( r0, v0 ) / sqrt_m ) * a * ( 1 - cos( x / sqrt_a ) )
        out += _r0 * sqrt_a * sin( x / sqrt_a )
        out -= sqrt_m * t

        return out
    
    return _func


def NL_grad( r0, v0, mu, a, t ):

    sqrt_m = sqrt( mu )
    sqrt_a = sqrt( a )

    _r0 = norm( r0 )

    def _grad( x ):

        out = 0.0
        out += a * ( 1 - cos( x / sqrt_a ) )
        out += ( dot( r0, v0 ) / sqrt_m ) * sqrt_a * sin( x / sqrt_a )
        out += _r0 * cos( x / sqrt_a )

        return out

    return _grad



if __name__ == "__main__":

    outF = zeros(2)
    outJ = zeros((2,2))

    r0 = array( [-18966.02307806,   1460.66653646,    843.31621802] )
    v0 = array( [-2.01566097, -2.90294796, -1.67601778] )

    mu = MU
    t  = 9113.471
    a  = 15000.0

    _r0 = 25990.316 
    _r1 = 17638.303 
    _r2 = 9353.01

    _func = NL_func( r0, v0, mu, a, t )
    _grad = NL_grad( r0, v0, mu, a, t )

    xS = newtonRaphson( _func, _grad, 0.0 )
    print(xS)

    _func = LM_func( _r0, _r1, _r2, outF )
    _jacb = LM_jacb( _r0, _r1, _r2, outJ )

    x0 = array([pi/4, pi/4])
    xS = levenbergMarquardt( _func, _jacb, x0 )
    print( xS )