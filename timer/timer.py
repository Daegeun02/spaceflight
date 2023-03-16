from threading import Thread

from time import sleep, time



class GlobalTim(Thread):
    '''
    To synchronize timer in satellite and ground,
    use same time source.
    '''


    def __init__(self, Hz=20, debug=False):

        super().__init__()

        self.daemon = True

        self.tim = 0

        if debug:
            self.dt = ( 0.001 / Hz ) 
        else:
            self.dt = ( 1 / Hz )

        self.debug = debug

        self.ticking = True


    def run(self):
        
        dt = self.dt

        t_start = time()

        if self.debug:

            while self.ticking:

                self.tim += dt * 1000

                sleep( dt )
        
        else:

            while self.ticking:

                self.tim = time() - t_start

                sleep( dt )