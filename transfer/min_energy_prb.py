## minimum energy problem

import cvxpy as cp

from numpy import zeros
from numpy import sqrt

from numpy.linalg import norm



def solve( r1, r2, v1, v2, args ):

    a = args["a"]
    m = args["mu"]
    H = args["H"]

    _r1 = norm( r1 )
    _r2 = norm( r2 )

    v = zeros(6)
    v[:3] = v1
    v[3:] = v2

    V1 = cp.Variable(3)
    V2 = cp.Variable(3)

    e = cp.Variable(3)

    ## object function
    err = 0
    err += cp.sum_squares( V1 - v1 )
    err += cp.sum_squares( V2 - v2 )

    ## constraints
    H = H.T / m

    b1 = e + r1 / _r1
    b2 = e + r2 / _r2

    _V1 = cp.norm( V1 )
    _V2 = cp.norm( V2 )

    E1 = sqrt( m * ( ( 2 / _r1 ) - 1 / a ) )
    E2 = sqrt( m * ( ( 2 / _r2 ) - 1 / a ) )

    constraints = []
    constraints.append(
        H @ V1 == b1
    )
    constraints.append(
        H @ V2 == b2
    )


    prob = cp.Problem( 
        objective=cp.Minimize( err ),
        constraints=constraints
    )

    print( prob.solve() )
    print( e.value )
    print( V1.value )
    print( V2.value )

    print( norm( V1.value ) - E1 )
    print( norm( V2.value ) - E2 )



if __name__ == "__main__":
    from numpy import array
    from numpy import cross
    from numpy import sqrt


    GRAVCONST = 6.674e-11

    EARTHMASS = 5.9742e24

    r1 = array([7000,   0,0])
    r2 = array([   0,8000,0])

    h = cross( r1, r2 )

    H = zeros((3,3))
    H[0,1] =  h[2]
    H[0,2] = -h[1]
    H[1,0] = -h[2]
    H[1,2] =  h[0]
    H[2,0] =  h[1]
    H[2,1] = -h[0]

    a  = 7000
    e  = 0.1
    mu = GRAVCONST * EARTHMASS
    p  = a * ( 1 - ( e ** 2 ) )

    args = {
        "a" : a,
        "e" : e,
        "mu": mu,
        "p" : p,
        "H" : H
    }

    v1 = zeros(3)
    v1[1] = sqrt( mu / p ) * ( e + 1 )
    v2 = zeros(3)
    v2[0] = sqrt( mu / p ) * ( -1 )

    solve( r1, r2, v1, v2, args )