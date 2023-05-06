from JAX_optimizer import newtonRaphson

from jax import jacobian
from jax import jacfwd

from jax.numpy import array
from jax.numpy import cos, sin
from jax.numpy import sqrt
from jax.numpy import dot

from jax.numpy.linalg import norm



def UF_func( configs ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    sqrt_m = sqrt( m )

    _r0 = norm( r0 )

    def _func( x , t=t, a=a, v0=v0 ):

        sqrt_a = sqrt( a )

        out = 0.0
        out += a * ( x - sqrt_a * sin( x / sqrt_a ) )
        out += ( dot( r0, v0 ) / sqrt_m ) * a * ( 1 - cos( x / sqrt_a ) )
        out += _r0 * sqrt_a * sin( x / sqrt_a )
        out -= sqrt_m * t

        return out
    
    return _func


def UF_grad( configs ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    sqrt_m = sqrt( m )

    _r0 = norm( r0 )

    def _grad( x ):

        sqrt_a = sqrt( a )

        out = 0.0
        out += a * ( 1 - cos( x / sqrt_a ) )
        out += ( dot( r0, v0 ) / sqrt_m ) * sqrt_a * sin( x / sqrt_a )
        out += _r0 * cos( x / sqrt_a )

        return out

    return _grad


def FG_expr( configs ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]

    sqrt_a = sqrt( a )
    sqrt_m = sqrt( m )

    _r0 = norm( r0 )

    def _func( x, t ):

        f = 1 - ( a / _r0 ) * ( 1 - cos( x / sqrt_a ) )
        g = t - ( a / sqrt_m ) * ( x - sqrt_a * sin( x / sqrt_a ) )

        r = f * r0 + g * v0

        _r = 1
        _r += ( dot( r0, v0 ) / sqrt( m * a ) ) * sin( x / sqrt( a ) )
        _r += ( ( norm( r0 ) / a ) - 1 ) * cos( x / sqrt( a ) )
        _r *= a 

        fdot = (-1) * ( ( sqrt_a * sqrt_m ) / ( _r0 * _r ) ) * sin( x / sqrt_a )
        gdot = 1 - ( a / _r ) * ( 1 - cos( x / sqrt_a ) )

        v = fdot * r0 + gdot * v0

        return array( [ r, v ] ).reshape(-1)

    return _func


def FG_back( configs ):

    r0 = configs["r0"]
    v0 = configs["v0"]

    def solution( x, t ):

        _func = FG_expr( r0, v0, configs )

        return _func( x, t )

    return jacfwd( solution, argnums=(0,1) )