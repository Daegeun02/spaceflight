from JAX_universal import UF_func

from JAX_optimizer import newtonRaphson

from jax.numpy import array



if __name__ == "__main__":

    configs = {
        "r0": array([10000,0,0]),
        "v0": array([0,-4,0]),
        "mu": 3e6,
        "a" : 10000,
        "t" : 3000
    }

    _func = UF_func( configs )

    print( _func( 10 ) )

    x = newtonRaphson( _func, 0.0 ) 

    print( x )

    print( _func( x ) )