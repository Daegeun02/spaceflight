from JAX_universal import UF_func
from JAX_universal import UF_back
from JAX_universal import FG_expr
from JAX_universal import FG_back

from JAX_optimizer import newtonRaphson

from jax.numpy import array
from jax.numpy import dot
from jax.numpy import sqrt, cos, sin

from jax.numpy.linalg import norm

import numpy as np



GRAVCONST = 6.674e-11

EARTHMASS = 5.9742e24

MU = GRAVCONST * EARTHMASS / ( 1000 ** 3 )

r0 = array( [-12944.83921337, -6562.56493754, -3788.89863326] )
v0 = array( [1.86069651, -2.71514808, -1.56759148] )
mu = MU
a  = 10000.0
t  = 3000.0



if __name__ == "__main__":

    configs = {
        "r0": r0,
        "v0": v0,
        "mu": mu,
        "a" : a,
        "t" : t
    }

    _UF_func = UF_func( configs )
    _UF_grad = UF_back( configs )

    print( _UF_func( 10 ) )

    x = newtonRaphson( _UF_func, 0.0 ) 

    print( x )

    print( _UF_func( x ) )
    print( _UF_grad( t ) )

    r = 1
    r += ( dot( r0, v0 ) / sqrt( mu * a ) ) * sin( x / sqrt( a ) )
    r += ( ( norm( r0 ) / a ) - 1 ) * cos( x / sqrt( a ) )
    r *= a 

    print( 'xdot', sqrt( mu ) / r )

    _FG_func = FG_expr( r0, v0, configs )
    _FG_jacb = FG_back( r0, v0, configs )

    rv   = _FG_func( x, t )
    prp_ = np.array( _FG_jacb( x, t ) ).T.reshape(-1,2)

    prpt = prp_[0:3,0] * _UF_grad( t ) + prp_[0:3,1] 

    print( rv )
    # print( prp_ )
    print( prpt )

    pfpx = (-1) * ( a / norm( r0 ) ) * sin( x / sqrt( a ) ) * ( 1 / sqrt( a ) )
    pgpx = ( a / sqrt( mu ) ) * ( cos( x / sqrt( a ) ) - 1 )

    # print( pfpx * r0[:] + pgpx * v0[:] )

    # print( prpt - ( pfpx * r0[:] * _UF_grad( t ) + pgpx * v0[:] * _UF_grad( t ) ) )

    print( '===' * 20 )

    f = 1 - ( a / norm( r0 ) ) * ( 1 - cos( x / sqrt( a ) ) )
    g = t - ( a / sqrt( mu ) ) * ( x - sqrt( a ) * sin( x / sqrt( a ) ) )

    rt = f * r0 + g * v0

    print( 'rt', norm( rt ) )

    print( 'r', r )

    fdot = (-1) * ( ( sqrt( a ) * sqrt( mu ) ) / ( norm( r0 ) * r ) ) * sin( x / sqrt( a ) )
    gdot = 1 - ( a / r ) * ( 1 - cos( x / sqrt( a ) ) )

    vt = fdot * r0 + gdot * v0

    print( 'vt', vt )