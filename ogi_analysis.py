#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rebound

def main():
    sim = rebound.SimulationArchive("ogi_archive.bin")
    n_iter = len(sim)

    x = np.zeros((3,n_iter))
    y = np.zeros((3,n_iter))
    z = np.zeros((3,n_iter))
    a = np.zeros(n_iter)
    e = np.zeros(n_iter)

    for i in range(n_iter): 
        ps = sim[i].particles

        x[0][i] = ps[0].x
        y[0][i] = ps[0].y
        z[0][i] = ps[0].z

        x[1][i] = ps[1].x
        y[1][i] = ps[1].y
        z[1][i] = ps[1].z
        a[i] = ps[1].a
        e[i] = ps[1].e

        x[2][i] = ps[2].x
        y[2][i] = ps[2].y
        z[2][i] = ps[2].z

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x[0], y[0], z[0])
    ax.plot(x[1], y[1], z[1])
    ax.plot(x[2], y[2], z[2])
    plt.show()



if __name__=="__main__":
    main()
