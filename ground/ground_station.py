from threading import Thread

from numpy import zeros
from numpy import cos, sin

from time import sleep

from .visualizer import plot_trajectory



class GroundControlStation(Thread):
    '''
    Literally Ground Control Station

    Transform PQW coordinate system postion and velocity
    to ECI coordinate system.
    '''


    def __init__(self, satellite, globaltim, n=20000):

        super().__init__()

        self.daemon = True

        self.location = zeros((3,n))
        self.movement = zeros((3,n))

        self.attitude = zeros((3,3))

        self.satellite = satellite
        self.globaltim = globaltim

        self.tracking = True

        self.trackIdx = 0


    def run(self):

        satellite = self.satellite
        globaltim = self.globaltim

        AcN = satellite.AscendingNode
        OIc = satellite.OrbitalInclination
        AOP = satellite.ArgumentOfPerigee

        position = satellite.position
        velocity = satellite.velocity

        location = self.location
        movement = self.movement

        self.PQW2ECI( AcN, OIc, AOP )

        attitude = self.attitude

        tim = globaltim.tim
        dt  = globaltim.dt

        trackIdx = 0

        while self.tracking:

            location[:,trackIdx] = attitude @ position
            movement[:,trackIdx] = attitude @ velocity

            trackIdx += 1

            sleep( dt - 0.01 )

        self.trackIdx = trackIdx

        print(self.trackIdx)



    def join(self):

        super().join()

        plot_trajectory( self.location, self.trackIdx )


    def PQW2ECI(self, AcN, OIc, AOP):
        
        attitude = self.attitude

        cA, sA = cos( AcN ), sin( AcN )
        cO, sO = cos( OIc ), sin( OIc )
        cP, sP = cos( AOP ), sin( AOP )

        attitude[0,0] = cP * cA - sP * cO * sA
        attitude[1,0] = cP * sA + sP * cO * cA
        attitude[2,0] = sP * sO
        attitude[0,1] = (-1) * (sP * cA + cP * cO * sA)
        attitude[1,1] = cP * cO * cA - sP * sA
        attitude[2,1] = cP * sO
        attitude[0,2] = sO * sA
        attitude[1,2] = (-1) * sO * cA
        attitude[2,2] = cO