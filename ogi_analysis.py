#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rebound

def plot_sim(timetimes, semis, ecc):
    fig = plt.figure(figsize=(10,4))
    axs = fig.subplots(1,2)
    for times,a,e in zip(timetimes,semis,ecc):
        axs[0].plot(times,a)
        axs[1].plot(times,e)
    plt.savefig('ogi_all.png')

    fig = plt.figure(figsize=(10,4))
    axs = fig.subplots(1,2)
    axs[0].hist([el[-1] for el in semis])
    axs[1].hist([el[-1] for el in ecc])
    plt.savefig('hists.png')

def main():
    sims=[]
    n_sims=100

    timetimes=[]
    semis=[]
    ecc=[]

    particle_number = 1 # corresponds to the planet

    for i in range(n_sims):
        print('Reading in simulation #',i)
        sim = rebound.SimulationArchive('ogi_'+str(i)+'.bin')

        n_iter = len(sim)
        times = np.zeros(n_iter)
        a = np.zeros(n_iter)
        e = np.zeros(n_iter)

        for j in range(n_iter):
            ps = sim[j].particles

            times[j] = sim[j].t
            a[j] = ps[particle_number].a
            e[j] = ps[particle_number].e

        timetimes.append(times)
        semis.append(a)
        ecc.append(e)

    plot_sim(timetimes,semis,ecc)

if __name__=="__main__":
    main()
