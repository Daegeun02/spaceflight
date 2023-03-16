## LAMBERT PROBLEM
from geometry     import GRAVCONST, EARTHMASS

from optimization import LevenbergMarquardt

from numpy import sqrt, cos



class LambertProblem:
    '''
    Calculate transfer orbit from r1 to r2.

    This function solve root finding problem 
    to get two major factor of transfer orbit.

    Solve root finding with Levenberg-Marquardt Algorithm

    From two major factor of transfer orbit,
    calculate six elements of orbit.
    '''

    LM_method = LevenbergMarquardt()

    def __init__(self):

        self.G = GRAVCONST
        self.M = EARTHMASS

        self.mu = GRAVCONST * EARTHMASS / ( 1000 ** 3 )


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

        '''
        Do Levenberg-Marquardt Algorithm
        '''

        '''
        Calculate six elements of orbit
        '''


def f1():
    pass


def f2():
    pass