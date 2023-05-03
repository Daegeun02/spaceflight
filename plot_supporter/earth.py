from geometry import EARTHRADS as Re

from numpy import linspace
from numpy import cos, sin
from numpy import deg2rad

from pandas import pd

from itertools import product



def draw_earth( ax ):
    # angles
    polars = linspace(0, 180, 19)
    azimuths = linspace(0, 360, 37)

    # points
    df = pd.DataFrame(product(polars, azimuths), columns=["azi", "polar"])
    df["x"] = df.apply(lambda x: cos(deg2rad(x[1]))*sin(deg2rad(x[0])), axis=1)
    df["y"] = df.apply(lambda x: sin(deg2rad(x[1]))*sin(deg2rad(x[0])), axis=1)
    df["z"] = df.apply(lambda x: cos(deg2rad(x[0])), axis=1)

    ax.plot_surface(df["x"].values.reshape((19, 37))*Re, 
                    df["y"].values.reshape((19, 37))*Re, 
                    df["z"].values.reshape((19, 37))*Re, 
                    ec="w", lw=0.2, ls=":",
                    cmap="viridis",
                    alpha=0.3)
    ax.set_box_aspect((1, 1, 1))