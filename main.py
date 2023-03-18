from execute import do_realtime
from execute import do_simulate

import matplotlib.pyplot as plt

from numpy import array



if __name__ == "__main__":

    trajectory, sub_trajectory = do_simulate(dt=10)

    trajectory = array( trajectory )
    sub_trajectory = array( sub_trajectory )

    print(trajectory.shape)

    plt.scatter( trajectory[:,0], trajectory[:,1], s=10 )
    plt.scatter( sub_trajectory[:,0], sub_trajectory[:,1], s=10 )
    
    plt.axis('equal')

    plt.show()