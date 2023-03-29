from numpy import cos, sin
from numpy import sqrt

from numpy.linalg import norm



def UF_func( args ):

    r0 = args["r0"]
    v0 = args["v0"]
    m  = args["mu"]
    a  = args["a"]
    t  = args["t"]

    sqrt_a = sqrt( a )
    sqrt_m = sqrt( m )

    def _func( x ):

        out = 0.0
        out += a * ( x - sqrt_a * sin( x / sqrt_a ) )
        out += ( r0 * v0 / sqrt_m ) * a * ( 1 - cos( x / sqrt_a ) )
        out += r0 * sqrt_a * sin( x / sqrt_m )
        out -= sqrt_m * t

        return out

    return _func


def UF_grad( args ):

    r0 = args["r0"]
    v0 = args["v0"]
    m  = args["mu"]
    a  = args["a"]

    sqrt_a = sqrt( a )
    sqrt_m = sqrt( m )

    def _grad( x ):

        out = 0.0
        out += a * ( 1 - cos( x / sqrt_a ) )
        out += ( r0 * v0 / sqrt_m ) * sqrt_a * sin( x / sqrt_a )
        out += r0 * cos( x / sqrt_a )

        return out

    return _grad


def f_and_g_expression( x, r0, v0, args ):

    m = args["mu"]
    a = args["a"]
    t = args["t"]

    sqrt_a = sqrt( a )
    sqrt_m = sqrt( m )
    _r0 = norm( r0 )
    _v0 = norm( v0 )

    f = 1 - ( a / _r0 ) * ( 1 - cos( x / sqrt_a ) )
    g = t - ( a / sqrt_m ) * ( x - sqrt_a * sin( x / sqrt_a ) )

    r = f * r0 + g * v0

    _r = norm( r )

    fdot = (-1) * ( ( sqrt_a * sqrt_m ) / ( _r0 * _r ) ) * sin( x / sqrt_a )
    gdot = 1 - ( a / _r ) * ( 1 - cos( x / sqrt_a ) )

    v = fdot * r0 + gdot * v0

    return r, v 