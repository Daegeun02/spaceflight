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
        a = satellite.a
        e = satellite.e
        T = satellite.T
        o = satellite.o
        i = satellite.i
        w = satellite.w

        position = satellite.position
        velocity = satellite.velocity
        ## geometric model
        mu  = geometric.mu
        ## timer
        tim = globaltim.tim
        dt  = globaltim.dt

        p = a * ( 1 - ( e ** 2 ) )

        args = {
            "e" : e,
            "M" : -1,
            "n" : sqrt( mu / ( a ** 3 ) ),
            "E" : -1,
            "eC": sqrt( ( 1 + e ) / ( 1 - e ) ),
            "N" : -1,
            "p" : p,
            "pC": sqrt( mu / p )
        }

        print('initialize finished...')

        print('start estimate orbit')

        while self.estimating:

            step_1( T, args, globaltim.tim )

            step_2( args )

            step_3( args, position, velocity )

            sleep( dt - 0.01 )

        print('end')


    def join(self):

        super().join()