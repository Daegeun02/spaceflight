from coordinate import ECI2ORP

from lambert import LambertProblem

from numpy import cross
from numpy import arctan2

from numpy.linalg import norm



def step02( r_chs_0_ECI, r_trg_t_ECI, t_tof, mu ):

    ## direction of angular momentum vector
    H = cross( r_chs_0_ECI, r_trg_t_ECI )
    h = H / norm( H )

    ## define orbital plane
    o = arctan2(         -h[0], h[1] )
    i = arctan2( norm( h[:2] ), h[2] ) * (-1)

    R = ECI2ORP( o, i )

    r_chs_0_ORP = R @ r_chs_0_ECI
    r_trg_t_ORP = R @ r_trg_t_ECI

    theta_0 = arctan2( r_chs_0_ORP[1], r_chs_0_ORP[0] )
    theta_t = arctan2( r_trg_t_ORP[1], r_trg_t_ORP[0] )
    theta   = theta_t - theta_0

    ## define Lambert Problem
    LP = LambertProblem( mu )

    _r_chs_0 = norm( r_chs_0_ECI )
    _r_trg_t = norm( r_trg_t_ECI )

    t1 = 0
    t2 = t1 + t_tof

    ## solve Lambert Problem
    a = LP.solve( _r_chs_0, _r_trg_t, t1, t2, theta )

    print(a)

    return a, o, i, theta_t, theta_0, R