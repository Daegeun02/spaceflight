from threading import Thread

from .steps import step_1, step_2, step_3

from math import sqrt

from time import sleep



class TwoBodyOrbit(Thread):
    '''
    Estimate satellite's position and velocity in pqw coordinate.

    This function has three step's to calculate position and velocity.
    step 1. estimate eccentric anomaly
    step 2. calculate true anomaly
    step 3. calculate position and velocity
    '''


    def __init__(self, satellite, geometric, globaltim):

        super().__init__()

        self.daemon = True

        self.satellite = satellite
        self.geometric = geometric
        self.globaltim = globaltim

        self.estimating = True


    def run(self):

        print('initialize...')
        ## satellite object
        satellite = self.satellite
        geometric = self.geometric
        globaltim = self.globaltim
        ## Classical Orbital Elements
        SMA = satellite.SemiMajorAxis
        Ecc = satellite.Eccentricity
        PrP = satellite.PerigeePassage
        AcN = satellite.AscendingNode
        OIc = satellite.OrbitalInclination
        AOP = satellite.ArgumentOfPerigee

        position = satellite.position
        velocity = satellite.velocity
        ## geometric model
        mu  = geometric.mu
        ## timer
        tim = globaltim.tim
        dt  = globaltim.dt

        SemRec = SMA * ( 1 - Ecc ** 2 )

        args = {
            "Ecc"   : Ecc,
            "MeaAnm": -1,
            "AngFrq": sqrt( mu / ( SMA ** 3 ) ),
            "EccAnm": -1,
            "EccCef": sqrt( ( 1 + Ecc ) / ( 1 - Ecc ) ),
            "TruAnm": -1,
            "SemRec": SemRec,
            "SemCef": sqrt( mu / SemRec )
        }

        print('initialize finished...')

        print('start estimate orbit')

        while self.estimating:

            step_1( PrP, args, globaltim.tim )

            step_2( args )

            step_3( args, position, velocity )

            sleep( dt )

        print('end')


    def join(self):

        super().join()