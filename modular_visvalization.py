import numpy as np
import matplotlib
# matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib.colors import Colormap, LightSource
import matplotlib.cm as cm
import os
import pyvista as pv

def plot_nodes(nodes, size=5):
    xx = nodes[:, 0]
    yy = nodes[:, 1]
    zz = nodes[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(nodes.shape[0]):
        ax.scatter3D(xx[i], yy[i], zz[i], color='red', marker='.', s=size)

    return fig
