## derivatives of 'r' and 'nu'
from numpy import sqrt
from numpy import sin, cos



def deriv_r( args ):

    P = args["p"]
    M = args["mu"]
    N = args["nu"]
    E  = args["ECC"]

    def dr_dt( t, r ):

        return sqrt( M / P ) * E * sin( N )

    return dr_dt


def deriv_nu( args ):

    P = args["p"]
    M = args["mu"]
    N = args["nu"]
    E = args["ECC"]

    P3 = P ** 3

    def dnu_dt( t, nu ):

        return sqrt( M / P3 ) * ( ( 1 + E * cos( nu ) ) ** 2 )

    return dnu_dt


if __name__ == "__main__":

    args = {
        "p" : 100,
        "mu": 32,
        "nu": 0.8,
        "ECC" : 0.1
    }

    dr_dt = deriv_r( args )

    print( dr_dt( 1, 30 ) )