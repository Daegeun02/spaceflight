from .noname_sub_func import *

from ..two_body_problem.universal_form import *

from geometry import MU

from math import sqrt

from numpy import zeros, eye

from numpy.linalg import norm



def NN_func( configs, out ):

    Tr0 = configs["r_trg_0_ECI"]       ## initial position of target
    Tv0 = configs["v_trg_0_ECI"]       ## initial velocity of target
    Cr0 = configs["r_chs_0_ECI"]       ## initial position of chaser
    Cv0 = configs["v_chs_0_ECI"]       ## initial velocity of chaser

    trg_a = configs["trg_a"]           ## semimajor axis of target

    Tvt = zeros( 6 )                ## final point of target
    Cvt = zeros( 6 )                ## final point of chaser

    trg_configs = {
        "r0": Tr0,
        "v0": Tv0,
        "mu": MU,
        "a": trg_a,
        "t": -1
    }

    chs_configs = {
        "r0": Cr0,
        "v0": Cv0,
        "mu": MU,
        "a": -1,
        "t": -1
    }

    ## compile
    uf_trg_func = UF_func( trg_configs )
    uf_chs_func = UF_func( chs_configs )
    ## compile
    fg_trg_func = FG_func( trg_configs, Tvt )
    fg_chs_func = FG_func( chs_configs, Cvt )

    sqrt_2 = sqrt( 2 )

    def _func( X ):
        ## our optimizer
        xC  = X[0]      ## universal variable for chaser
        xT  = X[1]      ## universal varibale for target
        _t  = X[2]      ## time of flight
        Dv0 = X[3:6]    ## first impulse
        Dv1 = X[6:9]    ## second impulse
        _a  = X[9]      ## semimajor axis of transfer orbit

        ## main object function
        out[ 0 ] = norm( Dv0 ) / sqrt_2
        out[ 1 ] = norm( Dv1 ) / sqrt_2

        ## main constraint function
        out[ 2 ] = uf_trg_func( xT, t=_t,       v0=Tv0     )    ## energy equation for traget
        out[ 3 ] = uf_chs_func( xC, t=_t, a=_a, v0=Cv0+Dv0 )    ## energy equation for chaser

        fg_trg_func( xT, t=_t,       v0=Tv0     )   ## final state of target
        fg_chs_func( xC, t=_t, a=_a, v0=Cv0+Dv0 )   ## final state of chaser

        out[4:7] = Tvt[0:3] - Cvt[0:3]          ## position constraint
        out[7: ] = Tvt[3:6] - Cvt[3:6] - Dv1    ## velocity constraint

        return out

    return _func


def NN_jacb( configs, out ):

    Tr0 = configs["r_trg_0_ECI"]       ## initial position of target
    Tv0 = configs["v_trg_0_ECI"]       ## initial velocity of target
    Cr0 = configs["r_chs_0_ECI"]       ## initial position of chaser
    Cv0 = configs["v_chs_0_ECI"]       ## initial velocity of chaser

    trg_a = configs["trg_a"]           ## semimajor axis of target

    Tvt = zeros( 6 )                ## final point of target
    Cvt = zeros( 6 )                ## final point of chaser

    ## constants
    sqrt_m = sqrt( MU )
    _Cr0   = norm( Cr0 )
    _Tr0   = norm( Tr0 )

    _I = eye( 3 ) * (-1)

    trg_configs = {
        "r0": Tr0,
        "v0": Tv0,
        "mu": MU,
        "a": trg_a,
        "t": -1
    }

    chs_configs = {
        "r0": Cr0,
        "v0": Cv0,
        "mu": MU,
        "a": -1,
        "t": -1
    }

    trg_args = {
        "sqrt_a": sqrt( trg_a ),
        "sqrt_m": sqrt_m,
        "cx"    : 0,
        "sx"    : 0
    }

    chs_args = {
        "sqrt_a": 0,
        "sqrt_m": sqrt_m,
        "cx"    : 0,
        "sx"    : 0
    }

    ## compile ##
    ## f and g expression
    chs_calG = calG( chs_configs, chs_args )
    ## target's partial derivatives
    trg_pEpx = pEpx( trg_configs, trg_args )
    trg_pEpt = pEpt( trg_configs, trg_args )
    trg_prpx = prpx( trg_configs, trg_args )
    trg_prpt = prpt( trg_configs, trg_args )
    trg_pvpx = pvpx( trg_configs, trg_args )
    trg_pvpt = pvpt( trg_configs, trg_args )

    ## chaser's partial derivatives
    chs_pEpx = pEpx( chs_configs, chs_args )
    chs_pEpt = pEpt( chs_configs, chs_args )
    chs_pEpa = pEpa( chs_configs, chs_args )
    chs_prpx = prpx( chs_configs, chs_args )
    chs_prpt = prpt( chs_configs, chs_args )
    chs_prpa = prpa( chs_configs, chs_args )
    chs_pvpx = pvpx( chs_configs, chs_args )
    chs_pvpt = pvpt( chs_configs, chs_args )
    chs_pvpa = pvpa( chs_configs, chs_args )
    #############

    def _jacb( X ):
        ## our optimizer
        xC  = X[0]      ## universal variable for chaser
        xT  = X[1]      ## universal varibale for target
        _t  = X[2]      ## time of flight
        Dv0 = X[3:6]    ## first impulse
        Dv1 = X[6:9]    ## second impulse
        _a  = X[9]      ## semimajor axis of transfer orbit

        sqrt_a = sqrt( _a )

        trg_cx = cos( xT / trg_args["sqrt_a"] )
        trg_sx = sin( xT / trg_args["sqrt_a"] )

        chs_cx = cos( xC / sqrt_a )
        chs_sx = sin( xC / sqrt_a )

        rC = 1.0
        rC += dot( Cr0, Cv0+Dv0 ) * chs_sx / ( sqrt_a * sqrt_m )
        rC += ( _Cr0 / _a - 1 ) * chs_cx
        rC *= _a

        rT = 1.0
        rT += dot( Tr0, Tv0 ) * trg_sx / ( sqrt_a * sqrt_m )
        rT += ( _Tr0 / trg_a - 1 ) * trg_cx
        rT *= trg_a

        ## update target args
        trg_args["cx"]     = trg_cx
        trg_args["sx"]     = trg_sx
        trg_args["radius"] = rT
        ## update chaser args
        chs_args["sqrt_a"] = sqrt_a
        chs_args["cx"]     = chs_cx
        chs_args["sx"]     = chs_sx
        chs_args["radius"] = rC

        ## make jacobian
        out[ 2 , 1 ] = trg_pEpx( xT, t=_t                   )
        out[ 2 , 2 ] = trg_pEpt( xT, t=_t                   )

        out[ 3 , 0 ] = chs_pEpx( xC, t=_t, a=_a, v0=Cv0+Dv0 )
        out[ 3 , 2 ] = chs_pEpt( xC, t=_t, a=_a, v0=Cv0+Dv0 )
        out[ 3 , 9 ] = chs_pEpa( xC, t=_t, a=_a, v0=Cv0+Dv0 )

        out[4:7, 0 ] = chs_prpx( xC, t=_t, a=_a, v0=Cv0+Dv0 ) * (-1)
        out[4:7, 1 ] = trg_prpx( xT, t=_t,                  )
        out[4:7, 2 ] = trg_prpt( xT, t=_t                   ) - chs_prpt( xC, t=_t, a=_a, v0=Cv0+Dv0 )
        out[4:7, 9 ] = chs_prpa( xC, t=_t, a=_a, v0=Cv0+Dv0 ) * (-1)

        out[ 3 ,3:6] = ( sqrt_a / sqrt_m ) * chs_sx * Cr0

        g, gdot = chs_calG( xC, t=_t, a=_a, v0=Cv0+Dv0 )
        out[4:7,3:6] = _I * g
        out[7: ,3:6] = _I * gdot

        out[7: , 0 ] = chs_pvpx( xC, t=_t, a=_a, v0=Cv0+Dv0 ) * (-1)
        out[7: , 1 ] = trg_pvpx( xT, t=_t,                  )
        out[7: , 2 ] = trg_pvpt( xT, t=_t                   ) - chs_pvpt( xC, t=_t, a=_a, v0=Cv0+Dv0 )
        out[7: , 9 ] = chs_pvpa( xC, t=_t, a=_a, v0=Cv0+Dv0 ) * (-1)

        out[ 0 ,3:6] = Dv0
        out[ 1 ,6:9] = Dv1
        out[7: ,6:9] = _I

        return out
    
    return _jacb