from coordinate import ECI2PQW

from geometry import MU

from numpy import arctan2, arctan, tan
from numpy import sin
from numpy import sqrt



def cross_check( O_orp, r_chs_0_ECI, r_trg_t_ECI ):

    a = O_orp["a"]
    e = O_orp["e"]
    o = O_orp["o"]
    i = O_orp["i"]
    w = O_orp["w"]
    T = O_orp["T"]

    R = ECI2PQW( o, i, w )

    r_chs_0_PQW = R @ r_chs_0_ECI
    r_trg_t_PQW = R @ r_trg_t_ECI

    N_chs = arctan2( r_chs_0_PQW[1], r_chs_0_PQW[0] )
    N_trg = arctan2( r_trg_t_PQW[1], r_trg_t_PQW[0] )

    eC = sqrt( ( 1 - e ) / ( 1 + e ) )

    E_chs = 2 * arctan( 
        eC * tan( N_chs / 2 )
    )

    E_trg = 2 * arctan( 
        eC * tan( N_trg / 2 )
    )

    M_chs = E_chs - e * sin( E_chs )
    M_trg = E_trg - e * sin( E_trg )

    n = sqrt( MU / ( a ** 3 ) )

    t_chs = ( M_chs / n ) + T
    t_trg = ( M_trg / n ) + T

    return t_trg - t_chs