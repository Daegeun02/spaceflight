## LAMBERT PROBLEM
from optimization import levenbergMarquardt

from numpy import sqrt, cos, sin
from numpy import pi
from numpy import zeros

from numpy.random import rand



class LambertProblem:
    '''
    Calculate transfer orbit from r1 to r2.

    This function solve root finding problem 
    to get two major factor of transfer orbit.

    Solve root finding with Levenberg-Marquardt Algorithm

    From two major factor of transfer orbit,
    calculate six elements of orbit.
    '''

    def __init__(self):
        pass


    def solve(self, r1, r2, t1, t2, theta, mu):
        '''
        Literally solve LAMBERT PROBLEM

        parameter  <br>
        r1: vector from focus to starting point <br>
        r2: vector from focus to arriving point  <br>
        t1: time when starting transfer <br>
        t2: time when arriving transfer <br>
        theta: the angle between r1 and r2 vector
        '''
        c = sqrt( 
            ( r1 ** 2 ) + ( r2 ** 2 ) - 2 * r1 * r2 * cos( theta )
        )

        s = ( r1 + r2 + c ) / 2
        q = ( sqrt( r1 * r2 ) / s ) * cos( theta / 2 )
        T = ( 1 / s ) * sqrt( 8 * mu / s ) * ( t2 - t1 )

        args = {
            "T": T,
            "q": q
        }

        f = zeros(2)
        J = zeros((2,2))

        x0 = zeros(2) + pi

        func = LP_func( args, f ) 
        jacb = LP_jacb( args, J )

        ## solve with alpha, beta by levenberg-marquardt algorithm
        xS = levenbergMarquardt( func, jacb, x0 )

        a = s / ( 1 - cos( xS[0] ) )

        return a


def LP_func( args, out ):

    T = args["T"]
    q = args["q"]

    def _func( x ):

        a, b = x

        out[0] = sin( b / 2 ) - q * sin( a / 2 )
        out[1] = T * ( sin( a / 2 ) ** 3 ) - ( a - b - sin( a ) + sin( b ) )

        return out

    return _func


def LP_jacb( args, out ):

    T = args["T"]
    q = args["q"]

    def _jacb( x ):

        a, b = x

        ca2 = cos( a / 2 )
        sa2 = sin( a / 2 )

        out[0,0] = q * ca2 * (-0.5)

        out[1,0] = T * ( sa2 ** 2 ) * ca2 * (1.5) - 1 + cos( a )

        out[0,1] = cos( b / 2 ) * (0.5)

        out[1,1] = 1 - cos( b )

        return out

    return _jacb