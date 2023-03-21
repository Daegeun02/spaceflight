from numpy import zeros 
from numpy import deg2rad
from numpy import cos, sin
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

        pqw2eci = self.PQW2ECI( self.o, self.i, self.w )

        self.position[0] = self.a * ( 1 - self.e )

        self.velocity[1] = sqrt( m / p ) * ( self.e + 1 )

        self.state[:3] = pqw2eci @ self.position
        self.state[3:] = pqw2eci @ self.velocity


    def PQW2ECI(self, o, i, w):

        pqw2eci = zeros((3,3))

        co, so = cos( o ), sin( o )
        ci, si = cos( i ), sin( i )
        cw, sw = cos( w ), sin( w )

        pqw2eci[0,0] = cw * co - sw * ci * so
        pqw2eci[1,0] = cw * so + sw * ci * co
        pqw2eci[2,0] = sw * si
        pqw2eci[0,1] = (-1) * (sw * co + cw * ci * so)
        pqw2eci[1,1] = cw * ci * co - sw * so
        pqw2eci[2,1] = cw * si
        pqw2eci[0,2] = si * so
        pqw2eci[1,2] = (-1) * si * co
        pqw2eci[2,2] = ci

        return pqw2eci