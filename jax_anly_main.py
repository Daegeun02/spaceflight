from optimization import newtonRaphson

from transfer import UF_func, UF_grad
from transfer import FG_expr

from geometry import MU

from numpy import array, zeros
from numpy import dot
from numpy import sqrt
from numpy import cos, arccos
from numpy import sin

from numpy.linalg import norm



def backward( configs ):

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]
    r_trg_0_ECI = configs["r_trg_0_ECI"]
    v_trg_0_ECI = configs["v_trg_0_ECI"]

    mu = configs["mu"]
    a  = configs["a"]
    t  = configs["t"]

    r0 = r_trg_0_ECI
    v0 = v_trg_0_ECI

    def _func( t ):

        trg_configs = {
            "r0": r_trg_0_ECI,
            "v0": v_trg_0_ECI,
            "mu": mu,
            "a" : a,
            "t" : t
        }

        ## RT backward BEGIN ##
        func = UF_func( trg_configs )
        grad = UF_grad( trg_configs )

        x = newtonRaphson( func, grad, 0.0 )

        r_trg_t_ECI, v_trg_t_ECI = FG_expr( x, r0, v0, trg_configs )

        xdot = sqrt( mu ) / norm( r_trg_t_ECI )

        rdot = v_trg_t_ECI
        ## RT backward END ##

        r_SKM = array([
            [0, -r_chs_0_ECI[2], r_chs_0_ECI[1]],
            [r_chs_0_ECI[2], 0, -r_chs_0_ECI[0]],
            [-r_chs_0_ECI[1], r_chs_0_ECI[0], 0]
        ])

        ## AM backward BEGIN ## 
        H    = r_SKM @ r_trg_t_ECI
        mngH = norm( H )
        h    = H / mngH

        rt = r_trg_t_ECI.reshape(-1,1)

        hdot = zeros((3,3))
        hdot += mngH * r_SKM
        hdot -= ( ( ( r_SKM @ rt ) @ ( r_SKM.T @ r_SKM @ rt ).T ) / mngH )
        hdot /= ( mngH ** 2 )
        hdot = hdot @ rdot
        ## AM backward END ##

        ## theta backward BEGIN ##
        r1 = norm( r_chs_0_ECI )
        r2 = norm( r_trg_t_ECI )

        X = dot( r_chs_0_ECI, r_trg_t_ECI ) / ( r1 * r2 )

        theta = arccos( X )

        thetadot = zeros(3)
        thetadot += r2 * r_chs_0_ECI
        thetadot -=  X * r1 * r_trg_t_ECI 
        thetadot /= ( r1 * ( r2**2 ) ) 
        thetadot = dot( thetadot, rdot )
        thetadot *= (-1) / sqrt( 1 - ( X**2 ) )
        ## theta backward END ##

        ## q backward BEGIN ##
        c = r1**2 + r2**2 - 2 * dot( r_chs_0_ECI, r_trg_t_ECI )
        s = (0.5) * ( r1 + r2 + c )

        q = ( sqrt( r1 * r2 ) / s ) * cos( theta/2 )

        cdot = r_trg_t_ECI - ( 2 * r_chs_0_ECI )
        sdot = dot( (0.5) * ( ( r_trg_t_ECI / r2 ) + cdot ), rdot )

        qdot = ( sqrt( r1 * r2 ) / s ) * ( -sin( theta/2 ) * (0.5) )
        qdot *= thetadot
        qdot += (-1) * ( sqrt( r1 * r2 ) / ( s**2 ) ) * cos( theta/2 ) * sdot
        ## q backward END ##

        ## T backward BEGIN ##
        T = sqrt( ( 8 * mu ) / ( s**3 ) ) * t

        Tdot = (-1.5) * sqrt( 8 * mu ) * s**(-5/2) * t
        Tdot *= sdot
        Tdot += sqrt( ( 8 * mu ) / ( s**3 ) )
        ## T backward END ##

        return h, hdot

    return _func


if __name__ == "__main__":
    print( '<<< NUMPY >>>' )

    r_chs_0_ECI = array( [-1449.26847138, 6569.08693194, 0.        ] )
    v_chs_0_ECI = array( [-7.78632738, -0.97724683, 0.        ] )
    r_trg_0_ECI = array( [-12944.83921337, -6562.56493754, -3788.89863326] )
    v_trg_0_ECI = array( [1.86069651, -2.71514808, -1.56759148] )

    mu = MU
    a  = 10000.0
    t  = 4000.0

    configs = {
        "r_chs_0_ECI": r_chs_0_ECI,
        "v_chs_0_ECI": v_chs_0_ECI,
        "r_trg_0_ECI": r_trg_0_ECI,
        "v_trg_0_ECI": v_trg_0_ECI,

        "mu": mu,
        "a" : a,
        "t" : t
    }

    _grad = backward( configs )

    T, Tdot = _grad( t )

    print( 'T    =>', T )
    print( 'Tdot =>', Tdot )