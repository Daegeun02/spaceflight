from threading import Thread

from numpy import eye, zeros
from numpy import cos, sin

from time import sleep



GRAVCONST = 6.674e-11

EARTHMASS = 5.9742e24

class EARTH(Thread):


    def __init__(self, theta_0, globaltim):

        super().__init__()

        self.daemon = True

        self.mu = GRAVCONST * EARTHMASS  / ( 1000 ** 3 )

        self.GRAVITY = -1

        self.RotVelo = 464 / 1000 ## km/s

        self.ECI2ECEF  = eye( 3 )
        self.dECI2ECEF = zeros((3,3))
        self.theta_T   = theta_0
        self.d_theta   = self.Rotvelo * globaltim.dt

        self.rotating = True

        self.globaltim = globaltim


    def run(self):

        globaltim = self.globaltim

        ECI2ECEF  = self.ECI2ECEF
        dECI2ECEF = self.dECI2ECEF

        RotVelo = self.RotVelo

        theta_T = self.theta_T
        d_theta = self.d_theta

        tim = globaltim.tim
        dt  = globaltim.dt

        while self.rotating:

            self.rotation(
                ECI2ECEF,
                dECI2ECEF,
                RotVelo,
                theta_T,
                d_theta
            )

            self.theta_T = theta_T

            sleep( dt )


    def rotation(self, ECI2ECEF, dECI2ECEF, RotVelo, theta_T, d_theta):

        theta_T += d_theta

        ctheta_T = cos( theta_T )
        stheta_T = sin( theta_T )

        ECI2ECEF[0,0] = ctheta_T
        ECI2ECEF[1,0] = stheta_T * ( -1 )

        ECI2ECEF[0,1] = stheta_T
        ECI2ECEF[1,1] = ctheta_T

        dECI2ECEF[0,0] = stheta_T * RotVelo * ( -1 )
        dECI2ECEF[1,0] = ctheta_T * RotVelo * ( -1 )

        dECI2ECEF[0,1] = ctheta_T * RotVelo
        dECI2ECEF[1,1] = stheta_T * RotVelo * ( -1 )