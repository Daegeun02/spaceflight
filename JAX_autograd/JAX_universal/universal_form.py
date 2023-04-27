from jax.numpy import array
from jax.numpy import cos, sin
from jax.numpy import sqrt
from jax.numpy import dot



def norm( x ):

    return sqrt( dot( x, x ) )


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