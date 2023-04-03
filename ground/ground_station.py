from threading import Thread

from coordinate import ECI2PQW

from numpy import zeros

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

        self.satellite = satellite
        self.globaltim = globaltim

        self.tracking = True

        self.trackIdx = 0


    def run(self):

        satellite = self.satellite
        globaltim = self.globaltim

        o = satellite.o
        i = satellite.i
        w = satellite.w

        position = satellite.position
        velocity = satellite.velocity

        location = self.location
        movement = self.movement

        R = ECI2PQW( o, i, w )

        tim = globaltim.tim
        dt  = globaltim.dt

        trackIdx = 0

        while self.tracking:

            location[:,trackIdx] = R.T @ position
            movement[:,trackIdx] = R.T @ velocity

            trackIdx += 1

            sleep( dt - 0.01 )

        self.trackIdx = trackIdx


    def join(self):

        super().join()

        plot_trajectory( self.location, self.trackIdx )