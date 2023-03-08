from threading import Thread

from optimization import newtonRaphson

from numpy import sin, cos, tan, arctan

from math import sqrt

from time import sleep



class TwoBodyOrbit(Thread):


    def __init__(self, satellite, geometric, globaltim):

        super().__init__()

        self.daemon = True

        self.satellite = satellite
        self.geometric = geometric
        self.globaltim = globaltim

        self.estimating = True


    def run(self):
        '''
        Estimate satellite's position and velocity in pqw coordinate.

        This function has three step's to calculate position and velocity.
        step 1. estimate eccentric anomaly
        step 2. calculate true anomaly
        step 3. calculate position and velocity
        '''

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

        step_1 = self.step_1
        step_2 = self.step_2
        step_3 = self.step_3

        estimating = self.estimating

        print('start estimate orbit')

        while estimating:

            step_1( PrP, args, tim )

            step_2( args )

            step_3( args, position, velocity )

            sleep( dt )


    def join(self):

        super().join()


    def step_1(self, PrP, args, tim):
        
        ## Eccentric Anomaly
        EccAnm = 0
        ## Mean Anomaly
        args["MeaAnm"] = args["AngFrq"] * ( tim - PrP )

        args["EccAnm"] = newtonRaphson( func, grad, EccAnm, args )

    
    def step_2(self, args):

        EccAnm = args["EccAnm"]
        EccCef = args["EccCef"]

        args["TruAnm"] = arctan( 
            EccCef * tan( EccAnm / 2 )
        )

    
    def step_3(self, args, position, velocity):

        Ecc    = args["Ecc"]
        TruAnm = args["TruAnm"]
        SemRec = args["SemRec"]
        SemCef = args["SemCef"]

        cTruAnm = cos( TruAnm )
        sTruAnm = sin( TruAnm )

        r = SemRec / ( 1 + Ecc * cTruAnm ) 

        position[0] = r * cTruAnm
        position[1] = r * sTruAnm

        velocity[0] = SemCef * sTruAnm * ( -1 )
        velocity[1] = SemCef * ( Ecc + cTruAnm )

        
## function that find root => estimate Eccentric Anomaly
def func( EccAnm, args ):

    Ecc    = args["Ecc"]
    MeaAnm = args["MeaAnm"]

    return ( EccAnm - Ecc * sin(EccAnm) - MeaAnm )


def grad( EccAnm, args ):

    Ecc = args["Ecc"]

    return ( 1 - Ecc * cos(EccAnm) )