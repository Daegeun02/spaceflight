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

        self.dt = ( 1 / Hz ) 

        self.debug = debug

        self.ticking = True


    def run(self):
        
        dt = self.dt

        t_start = time()

        if self.debug:

            while self.ticking:

                self.tim += dt

                sleep( dt / 1000 )
        
        else:

            while self.ticking:

                self.tim = time() - t_start

                sleep( dt )