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

        o = satellite.o
        i = satellite.i
        w = satellite.w

        position = satellite.position
        velocity = satellite.velocity

        location = self.location
        movement = self.movement

        self.PQW2ECI( o, i, w )

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


    def join(self):

        super().join()

        plot_trajectory( self.location, self.trackIdx )


    def PQW2ECI(self, o, i, w):
        
        attitude = self.attitude

        co, so = cos( o ), sin( o )
        ci, si = cos( i ), sin( i )
        cw, sw = cos( w ), sin( w )

        attitude[0,0] = cw * co - sw * ci * so
        attitude[1,0] = cw * so + sw * ci * co
        attitude[2,0] = sw * si
        attitude[0,1] = (-1) * (sw * co + cw * ci * so)
        attitude[1,1] = cw * ci * co - sw * so
        attitude[2,1] = cw * si
        attitude[0,2] = si * so
        attitude[1,2] = (-1) * si * co
        attitude[2,2] = ci