from execute import do_realtime
from execute import do_simulate

import matplotlib.pyplot as plt

from numpy import array



if __name__ == "__main__":

    trajectory = do_simulate(dt=0.05)

    trajectory = array( trajectory )

    print(trajectory.shape)

    plt.scatter( trajectory[:,0], trajectory[:,1], s=0.5 )

    plt.scatter( 0, 0, s=30 )
    
    plt.axis('equal')

    plt.grid()

    plt.show()