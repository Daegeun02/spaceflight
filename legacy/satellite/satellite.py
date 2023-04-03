from coordinate import ECI2PQW

from numpy import zeros 
from numpy import deg2rad
from numpy import sqrt



class Satellite:


    def __init__(self):

        self.a = 6600 + 600             ## semimajor axis
        self.e = 0.5                    ## eccentricity
        self.T = 0                      ## perigee passage
        self.o = deg2rad( 0.0 )         ## ascending node
        self.i = deg2rad( 0.0 )         ## orbital inclination
        self.w = deg2rad( 0.0 )         ## argument of perigee

        ## pqw coordinate
        self.position = zeros(3)
        self.velocity = zeros(3)

        self.state = zeros(6)

        self.period = 1e8

    
    def init_state(self, args):

        m = args["mu"]
        p = args["p"]

        R = ECI2PQW( self.o, self.i, self.w )

        self.position[0] = self.a * ( 1 - self.e )

        self.velocity[1] = sqrt( m / p ) * ( self.e + 1 )

        self.state[:3] = R.T @ self.position
        self.state[3:] = R.T @ self.velocity