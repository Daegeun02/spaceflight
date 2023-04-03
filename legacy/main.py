from legacy.execute import do_realtime
from legacy.execute import do_simulate

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from numpy import array

from cvxpy import Variable



if __name__ == "__main__":

    trajectory = do_simulate(dt=0.1, dnm=1)

    trajectory = array( trajectory )

    print(trajectory.shape)

    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    ax.plot3D( 
        trajectory[:,0], 
        trajectory[:,1], 
        trajectory[:,2]
    )

    ax.axis('equal')

    ax.set_xlabel( 'x-inertia' )
    ax.set_ylabel( 'y-inertia' )
    
    plt.show()