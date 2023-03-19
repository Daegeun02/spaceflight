from numpy import zeros 
from numpy import deg2rad



class Satellite:


    def __init__(self):

        self.a = 6600                   ## semimajor axis
        self.e = 0.5                    ## eccentricity
        self.T = 0                      ## perigee passage
        self.o = deg2rad( 0.0 )         ## ascending node
        self.i = deg2rad( 0.0 )         ## orbital inclination
        self.w = deg2rad( 0.0 )         ## argument of perigee

        self.r = self.a * ( 1 - self.e )        ## distance from focus
        self.N = 0                              ## true anomaly

        self.R_at_pqw = {
            "r": self.a * ( 1- self.e ),
            "N": 0
        }

        self.position = zeros(3)
        self.velocity = zeros(3)

        self.period = 1e8