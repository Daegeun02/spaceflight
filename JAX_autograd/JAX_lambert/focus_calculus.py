from jax.numpy import zeros
from jax.numpy import cos, sin



def FA_func( _r0, _r1, _r2 ):

    def _func( x ):

        out = zeros(2)

        a, b = x

        out = out.at[0].set( _r1 * cos( a ) + _r2 * cos( b ) - _r0 )
        out = out.at[1].set( _r1 * sin( a ) - _r2 * sin( b ) )

        return out

    return _func


def FA_jacb( _r0, _r1, _r2 ):

    def _jacb( x ):

        out = zeros((2,2))

        a, b = x

        out = out.at[0,0].set( _r1 * sin( a ) * ( -1 ) )
        out = out.at[1,0].set( _r1 * cos( a ) )
        out = out.at[0,1].set( _r2 * sin( b ) * ( -1 ) )
        out = out.at[1,1].set( _r2 * cos( b ) * ( -1 ) )

        return out
    
    return _jacb