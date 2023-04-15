from coordinate import ECI2PQW

from geometry import MU

from two_body_problem import step_1, step_2
from two_body_problem import func, grad

from numpy import arctan2
from numpy import sqrt



def true_anomaly_is( O_xxx, r_xxx_x_ECI ):

    a = O_xxx["a"]
    e = O_xxx["e"]

    p = a * ( 1 - ( e**2 ) )

    args = {
        "e" : e,
        "M" : -1,
        "n" : sqrt( MU / ( a ** 3 ) ),
        "E" : -1,
        "eC": sqrt( ( 1 + e ) / ( 1 - e ) ),
        "N" : -1,
        "p" : p,
        "pC": sqrt( MU / p ),
        "T" : 0
    }

    args["func"] = func( args )
    args["grad"] = grad( args )

    step_1( args, )


def true_anomaly( O_orp, r_xxx_x_ECI ):

    a = O_orp["a"]
    e = O_orp["e"]
    o = O_orp["o"]
    i = O_orp["i"]
    w = O_orp["w"]
    T = O_orp["T"]

    R = ECI2PQW( o, i, w )

    r_xxx_x_PQW = R @ r_xxx_x_ECI

    N_xxx = arctan2( r_xxx_x_PQW[1], r_xxx_x_PQW[0] )

    return N_xxx