import jax.numpy as jnp

from jax.numpy import exp 

from jaxopt import LevenbergMarquardt

from time import process_time


out = jnp.array( [0, 0], dtype=jnp.float32 )


def func( X, t ):
    
    x, y = X
    
    O1 = out.at[0].set( exp( y+t ) - x )
    O2 = O1.at[1].set( exp( x+t ) + y )
    
    return O2


init_params = jnp.array( [0.0, 0.0] )

t1 = process_time()

for i in range( 100 ):
    solution = LevenbergMarquardt( func ).run( init_params, 1 ).params

t2 = process_time()

print( t2 - t1 )

print( solution )