from optimization import newtonRaphson
from optimization import levenbergMarquardt

from lambert import LambertProblem
from lambert import FA_func, FA_jacb

from transfer import UF_func, UF_grad
from transfer import FG_expr

from geometry import MU

from numpy import array, zeros
from numpy import dot, eye
from numpy import sqrt, pi
from numpy import cos, arccos
from numpy import sin, arcsin

from numpy.linalg import norm

from numpy.random import rand



def backward( configs ):

    r_chs_0_ECI = configs["r_chs_0_ECI"]
    v_chs_0_ECI = configs["v_chs_0_ECI"]
    r_trg_0_ECI = configs["r_trg_0_ECI"]
    v_trg_0_ECI = configs["v_trg_0_ECI"]

    mu = configs["mu"]
    t  = configs["t"]

    a_trg  = configs["a_trg"]

    r0 = r_trg_0_ECI
    v0 = v_trg_0_ECI

    def _func( t ):

        trg_configs = {
            "r0": r_trg_0_ECI,
            "v0": v_trg_0_ECI,
            "mu": mu,
            "a" : a_trg,
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
        c = sqrt( r1**2 + r2**2 - 2 * dot( r_chs_0_ECI, r_trg_t_ECI ) )
        s = (0.5) * ( r1 + r2 + c )

        q = ( sqrt( r1 * r2 ) / s ) * cos( theta/2 )

        cdot = dot( ( r_trg_t_ECI -  r_chs_0_ECI ) / c, rdot )
        sdot = (0.5) * ( dot( ( r_trg_t_ECI / r2 ) , rdot ) + cdot )

        qdot = ( sqrt( r1 * r2 ) / s ) * ( -sin( theta/2 ) * (0.5) )
        qdot *= thetadot
        qdot += (-1) * ( sqrt( r1 * r2 ) / ( s**2 ) ) * cos( theta/2 ) * sdot
        qdot += sqrt( r1 / ( r2**3 ) ) / ( 2 * s ) * cos( theta/2 ) * dot( r_trg_t_ECI, rdot )
        ## q backward END ##

        ## T backward BEGIN ##
        T = sqrt( ( 8 * mu ) / ( s**3 ) ) * t

        Tdot = (-1.5) * sqrt( 8 * mu ) * s**(-5/2) * t
        Tdot *= sdot
        Tdot += sqrt( ( 8 * mu ) / ( s**3 ) )
        ## T backward END ##

        ## A backward BEGIN ##
        LP = LambertProblem()
        _a = LP.solve( r1, r2, 0, t, theta, mu )
        xS = LP.xS
        A  = xS[0]

        ca = cos( xS[0]/2 )
        sa = sin( xS[0]/2 )

        g    = q * sa
        fdot = 1 / sqrt( 1 - ( g**2 ) )

        _D = 0
        _D += (1.5) * T * (sa**2) * ca
        _D += fdot * q * ca
        _D -= 1
        _D += cos( xS[0] )
        _D -= cos( 2 * arcsin( g ) ) * fdot * q * ca

        _N = 0
        _N -= Tdot * (sa**3)
        _N -= 2 * fdot * qdot * sa
        _N += cos( 2 * arcsin( g ) ) * 2 * fdot * qdot * sa

        Adot = _N / _D
        ## A backward END ##

        ## a backward BEGIN ##
        _adot = 0
        _adot += ( s / ( ( 1 - cos( A ) )**2 ) ) * sin( A ) * Adot
        _adot += ( 1 / ( 1 - cos( A ) ) ) * sdot
        ## a backward END ##

        ## l backward BEGIN ##
        _r0 = norm( r_chs_0_ECI - r_trg_t_ECI )
        _r1 = 2 * _a - r1
        _r2 = 2 * _a - r2
        
        args = {
            "_r": _r0,
            "r1": _r1,
            "r2": _r2
        }

        f = zeros(2)
        J = zeros((2,2))

        func = FA_func( args, f )
        jacb = FA_jacb( args, J )

        x0 = array([pi/4, pi/4])
        L = levenbergMarquardt( func, jacb, x0 )
        l = _r1 * sin( L[0] )

        g    = ( _r1 / _r2 ) * sin( L[0] )
        gdot = ( 1 / sqrt( 1 - ( g**2 ) ) )

        _D = 0
        _D -= _r1 * sin( L[0] )
        _D -= _r1 * sin( L[0] ) * gdot * ( _r1 / _r2 ) * cos( L[0] )

        _N = 0
        _N -= cos( arcsin( g ) )
        _N += g * gdot * g
        _N *= 2 * _adot - ( dot( r_trg_t_ECI, v_trg_t_ECI ) / _r2 )
        _N += cdot

        ldot = 0
        ldot += r1 * cos( L[0] ) * ( _N / _D )
        ldot += 2 * _adot * sin( L[0] )
        ## l backward END ##

        ## d backward BEGIN ##
        d = ( r_trg_t_ECI - r_chs_0_ECI ) / c

        ddot = 0
        ddot += dot( ( eye(3) / c ), rdot )
        ddot -= ( d / c ) * cdot
        ## d backward END ##

        ## D backward BEGIN ##
        h_SKM = array([
            [0, -h[2], h[1]],
            [h[2], 0, -h[0]],
            [-h[1], h[0], 0]
        ])
        d_SKM = array([
            [0, -d[2], d[1]],
            [d[2], 0, -d[0]],
            [-d[1], d[0], 0]
        ])
        D = h_SKM @ d

        Ddot = h_SKM @ ddot - d_SKM @ hdot
        ## D backward END ##

        ## O backward BEGIN ##

        ## O backward END ##

        return x, A, l, xdot, Adot, ldot 
        # return A, _a, l, d, D, Adot, _adot, ldot, ddot, Ddot

    return _func


if __name__ == "__main__":
    print( '<<< NUMPY >>>' )

    r_chs_0_ECI = array( [2594.76851273, 8860.1442633,     0.        ] )
    v_chs_0_ECI = array( [-6.40773121,  2.33323034,  0.        ] )
    r_trg_0_ECI = array( [-18966.02307806,   1460.66653646,    843.31621802] )
    v_trg_0_ECI = array( [-2.01566097, -2.90294796, -1.67601778] )

    mu = MU
    t  = 9113.471

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

    _grad = backward( configs )

    x, A, l, xdot, Adot, ldot = _grad( t )

    print( 'val =>', x, A, l )
    print( 'dot =>', xdot, Adot, ldot )