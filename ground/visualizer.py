import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d



def plot_trajectory( location, trackIdx ):

    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    ax.plot3D( 
        location[0,1:trackIdx], 
        location[1,1:trackIdx], 
        location[2,1:trackIdx]
    )

    ax.set_xlabel( 'x-inertia' )
    ax.set_ylabel( 'y-inertia' )

    plt.show()


def plot_2_trajectory( location1, location2, trackIdx1, trackIdx2 ):

    fig = plt.figure()
    ax  = plt.axes(projection='3d')

    ax.plot3D( 
        location1[0,1:trackIdx1], 
        location1[1,1:trackIdx1], 
        location1[2,1:trackIdx1]
    )

    ax.plot3D( 
        location2[0,1:trackIdx2], 
        location2[1,1:trackIdx2], 
        location2[2,1:trackIdx2]
    )

    ax.set_xlabel( 'x-inertia' )
    ax.set_ylabel( 'y-inertia' )

    plt.show()