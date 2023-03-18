## derivatives of 'r' and 'nu'
from numpy import sqrt
from numpy import sin, cos



def deriv_r( args ):

    p = args["p"]
    m = args["mu"]
    N = args["N"]
    e = args["e"]

    def dr_dt( t, r ):

        return sqrt( m / p ) * e * sin( N )

    return dr_dt


def deriv_N( args ):

    p = args["p"]
    m = args["mu"]
    e = args["e"]

    p3 = p ** 3

    def dN_dt( t, N ):

        return sqrt( m / p3 ) * ( ( 1 + e * cos( N ) ) ** 2 )

    return dN_dt


if __name__ == "__main__":

    args = {
        "p" : 100,
        "mu": 32,
        "N": 0.8,
        "e" : 0.1
    }

    dr_dt = deriv_r( args )

    print( dr_dt( 1, 30 ) )