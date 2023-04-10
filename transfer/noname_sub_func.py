from math import sqrt
from math import cos, sin

from numpy import dot
from numpy import zeros

from numpy.linalg import norm



def pEpx( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    def _pEpx( x, t=t, a=a, v0=v0 ):

        sqrt_a = args["sqrt_a"]
        sqrt_m = args["sqrt_m"]

        cx = args["cx"]
        sx = args["sx"]

        out = 0.0
        out += a * ( 1 - cx )
        out += ( sqrt_a / sqrt_m ) * dot( r0, v0 ) * sx
        out += _r0 * cx

        return out

    return _pEpx


def pEpt( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    def _pEpt( x, t=t, a=a, v0=v0 ):

        sqrt_m = args["sqrt_m"]
        sqrt_a = args["sqrt_a"]
        radius = args["radius"]

        cx = args["cx"]
        sx = args["sx"]

        out = 0.0
        out += a * ( 1 - cx )
        out += ( sqrt_a / sqrt_m ) * dot( r0, v0 ) * sx
        out += _r0 * cx
        out /= radius
        out -= 1
        out *= sqrt_m

        return out
    
    return _pEpt


def pEpa( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    def _pEpa( x, t=t, a=a, v0=v0 ):

        sqrt_m = args["sqrt_m"]
        sqrt_a = args["sqrt_a"]
        
        cx = args["cx"]
        sx = args["sx"]

        out = 1.0
        out -= ( cx + ( x / ( 2 * sqrt_a ) ) * sx )
        out *= dot( r0, v0 ) / sqrt_m
        out += _r0 * ( ( 0.5 / sqrt_a ) * sx - ( x / ( 2 * sqrt_a ) ) * cx )
        out += x
        out -= (1.5) * sqrt_a * sx
        out += (0.5) * x * cx

        return out
    
    return _pEpa


def prpx( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    out = zeros( 3 )

    def _prpx( x, t=t, a=a, v0=v0 ):

        sqrt_m = args["sqrt_m"]
        sqrt_a = args["sqrt_a"]

        cx = args["cx"]
        sx = args['sx']

        pfpx = ( sqrt_a / _r0 ) * sx * (-1)
        pgpx = ( cx - 1 ) * a / sqrt_m

        out[:] = pfpx * r0 + pgpx * v0

        return out
    
    return _prpx


def prpt( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    out = zeros( 3 )

    def _prpt( x, t=t, a=a, v0=v0 ):

        sqrt_m = args["sqrt_m"]
        sqrt_a = args["sqrt_a"]
        radius = args["radius"]

        cx = args["cx"]
        sx = args["sx"]

        pfpt = ( sqrt_m * sqrt_a ) / ( _r0 * radius ) * sx * (-1)
        pgpt = 1 - ( a / radius ) * ( 1 - cx )

        out[:] = pfpt * r0 + pgpt * v0

        return out
    
    return _prpt


def prpa( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    out = zeros( 3 )

    def _prpa( x, t=t, a=a, v0=v0 ):

        sqrt_m = args["sqrt_m"]
        sqrt_a = args["sqrt_a"]

        cx = args["cx"]
        sx = args["sx"]

        pfpa = ( cx - 1 + ( x / ( 2 * sqrt_a ) ) * sx ) / _r0
        pgpa = ( (1.5) * sqrt_a * sx - (0.5) * x * ( 2 + cx ) ) / sqrt_m

        out[:] = pfpa * r0 + pgpa * v0

        return out
    
    return _prpa


def pvpx( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    out = zeros( 3 )

    def _pvpx( x, t=t, a=a, v0=v0 ):

        sqrt_m = args["sqrt_m"]
        sqrt_a = args["sqrt_a"]
        radius = args["radius"]

        cx = args["cx"]
        sx = args["sx"]

        prpx = ( dot( r0, v0 ) / sqrt_m ) * cx - ( ( _r0 - a ) / radius ) * sx

        pfpx = ( ( sqrt_a * sqrt_m ) / ( _r0 * radius**2 ) ) * prpx * sx - ( sqrt_m / ( _r0 * radius ) ) * cx
        pgpx = ( a / radius**2 ) * prpx * ( 1 - cx ) + ( sqrt_a / radius ) * sx

        out[:] = pfpx * r0 + pgpx * v0

        return out
    
    return _pvpx


def pvpt( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    _r0 = norm( r0 )

    out = zeros( 3 )

    def _pvpt( x, t=t, a=a, v0=v0 ):

        sqrt_m = args["sqrt_m"]
        sqrt_a = args["sqrt_a"]
        radius = args["radius"]

        cx = args["cx"]
        sx = args["sx"]

        prpt = ( dot( r0, v0 ) / radius ) * cx - ( ( _r0 - a ) / sqrt_a ) * ( sqrt_m / radius ) * sx

        pfpt = ( ( sqrt_a * sqrt_m ) / ( _r0 * radius**2 ) ) * prpt * sx - ( m / ( _r0 * radius**2 ) ) * cx
        pgpt = ( a / radius**2 ) * prpt * ( 1 - cx ) + ( ( sqrt_a * sqrt_m ) / radius**2 ) * sx

        out[:] = pfpt * r0 + pgpt * v0

        return out
    
    return _pvpt


def pvpa( configs, args ):

    r0 = configs["r0"]
    v0 = configs["v0"]
    m  = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    def _pvpa( x, t=t, a=a, v0=v0 ):

        out = 0.0

        return out
    
    return _pvpa