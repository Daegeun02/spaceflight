from jax_forward import forward

from JAX_optimizer import optimize

from jax import jacobian

from jax.numpy import array


print( '<<< JAX >>>' )

GRAVCONST = 6.674e-11
EARTHMASS = 5.9742e24
MU        = GRAVCONST * EARTHMASS / ( 1000 ** 3 )

r_chs_0_ECI = array( [2594.76851273, 8860.1442633,     0.        ] )
v_chs_0_ECI = array( [-6.40773121,  2.33323034,  0.        ] )
r_trg_0_ECI = array( [-18966.02307806,   1460.66653646,    843.31621802] )
v_trg_0_ECI = array( [-2.01566097, -2.90294796, -1.67601778] )

mu = MU
t  = 9000.0

a_chs = 10000.0
a_trg = 15000.0

configs = {
    "r_chs_0_ECI": r_chs_0_ECI,
    "v_chs_0_ECI": v_chs_0_ECI,
    "r_trg_0_ECI": r_trg_0_ECI,
    "v_trg_0_ECI": v_trg_0_ECI,

    "mu": mu,
    "t" : t,
    
    "a_chs": a_chs,
    "a_trg": a_trg
}

'''
[DeviceArray(1.6706686, dtype=float32), DeviceArray(11153.463, dtype=float32), DeviceArray(16373.818, dtype=float32), DeviceArray(17555.992, dtype=float32)]
12765.339
[DeviceArray(1.671028, dtype=float32), DeviceArray(10736.966, dtype=float32), DeviceArray(16395.225, dtype=float32), DeviceArray(17680.266, dtype=float32)]
12713.157
[DeviceArray(1.6691871, dtype=float32), DeviceArray(11102.987, dtype=float32), DeviceArray(16376.285, dtype=float32), DeviceArray(17571.46, dtype=float32)]
12756.48
[DeviceArray(1.6691054, dtype=float32), DeviceArray(10799.341, dtype=float32), DeviceArray(16391.867, dtype=float32), DeviceArray(17662.145, dtype=float32)]
12703.914
[DeviceArray(1.6711509, dtype=float32), DeviceArray(11167.447, dtype=float32), DeviceArray(16373.137, dtype=float32), DeviceArray(17551.684, dtype=float32)]
12762.283
'''

if __name__ == "__main__":
    _func = forward( configs )
    _grad = jacobian( _func )

    tS = optimize( _func, _grad, t )

    print( tS, _func( tS ) )