## LAMBERT PROBLEM
from optimization import levenbergMarquardt

from numpy import sqrt, cos, sin
from numpy import arctan2
from numpy import zeros
from numpy import cross

from numpy.linalg import norm



class LambertProblem:
    '''
    Calculate transfer orbit from r1 to r2.

    This function solve root finding problem 
    to get two major factor of transfer orbit.

    Solve root finding with Levenberg-Marquardt Algorithm

    From two major factor of transfer orbit,
    calculate six elements of orbit.
    '''

    def __init__(self, geometry):

        self.mu = geometry.mu


    def solve(self, r1, r2, t1, t2, theta):
        '''
        Literally solve LAMBERT PROBLEM

        parameter  <br>
        r1: distance from focus to starting point <br>
        r2: distance from focus to arriving point  <br>
        t1: time when starting transfer <br>
        t2: time when arriving transfer <br>
        theta: the angle between r1 and r2 vector
        '''

        mu = self.mu

        c = sqrt( 
            ( r1 ** 2 ) + ( r2 ** 2 ) - 2 * r1 * r2 * cos( theta )
        )

        s = ( r1 + r2 + c ) / 2
        q = ( sqrt( r1 * r2 ) / s ) * cos( theta / 2 )
        T = ( 1 / s ) * sqrt( 8 * mu / s ) * ( t2 - t1 )

        args = {
            "T": T,
            "q": q,
            "f": zeros(2),
            "J": zeros((2,2))
        }

        x0 = zeros( 2 )

        ## solve with alpha, beta by levenberg-marquardt algorithm
        xS = levenbergMarquardt( func, jacb, x0, args )

        '''
        Calculate six elements of orbit
        '''
        orbital_element = {}

        a = s / ( 1 - cos( xS[0] ) )

        r1_ECI = -1
        r2_ECI = -1

        H = cross( r1_ECI, r2_ECI )

        o = arctan2( -H[0]        , H[1] )
        i = arctan2( norm( H[:2] ), H[2] )

        orbital_element["a"] = a
        orbital_element["o"] = o
        orbital_element["i"] = i


def func( x, args ):

    a, b = x

    T = args["T"]
    q = args["q"]
    f = args["f"]

    f[0] = sin( b / 2 ) - q * sin( a / 2 )
    f[1] = T * ( sin( a / 2 ) ** 3 ) - ( a - b - sin( a ) + sin( b ) )


def jacb( x, args ):
    
    a, b = x

    T = args["T"]
    q = args["q"]
    J = args["J"]

    ca2 = cos( a / 2 )
    sa2 = sin( a / 2 )

    J[0,0] = q * ca2 * (-0.5)

    J[1,0] = T * ( sa2 ** 2 ) * ca2 * (1.5) - 1 + cos( a )

    J[0,1] = ca2 * (0.5)

    J[1,1] = 1 - cos( b )