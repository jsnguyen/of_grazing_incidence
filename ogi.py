#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rebound

# may also be interpreted as random vectors that uniformly sample the surface of the sphere
def random_points_on_sphere(n_points):

    theta = np.zeros(n_points)
    phi = np.zeros(n_points)

    for i in range(n_points):
        u = np.random.uniform()
        v = np.random.uniform()

        theta[i] = np.arccos(2*v-1) # inclination
        phi[i] = 2*np.pi*u # azimuth

    return theta, phi

def get_cartesian_coordinates(coords):
    rho = coords[0,:]
    theta = coords[1,:]
    phi = coords[2,:]

    x = rho*np.sin(theta)*np.cos(phi)
    y = rho*np.sin(theta)*np.sin(phi)
    z = rho*np.cos(theta)

    return x,y,z


def random_points_on_circle(n_points):

    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)

    for i in range(n_points):
        phi = 2*np.pi*np.random.uniform() # azimuth

        x[i] = np.cos(phi)
        y[i] = np.sin(phi)

    return np.vstack((x,y,z))

def rotate_circle(coords, theta, phi):

    Ry = np.array([[np.cos(theta), 0 , np.sin(theta)],
                   [0,1,0],
                   [-np.sin(theta), 0 ,np.cos(theta)]])

    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])

    res = np.matmul(Rz,np.matmul(Ry,coords))

    return res

def main():
    sim = rebound.Simulation()
    sim.add(m=1)
    sim.add(m=1e-3, a=1)

    sim.move_to_com()
    ps = sim.particles

    n_iter = 100000

    year = 2.*np.pi # One year in units where G=1
    times = np.linspace(0,70*year, n_iter)

    x = np.zeros((2,n_iter))
    y = np.zeros((2,n_iter))
    z = np.zeros((2,n_iter))

    '''
    for i,time in enumerate(times):
        sim.integrate(time)
        x[0][i] = ps[0].x
        y[0][i] = ps[0].y
        z[0][i] = ps[0].z
        x[1][i] = ps[1].x
        y[1][i] = ps[1].y
        z[1][i] = ps[1].z

    sim.status()

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    ax.set_xlim([-6,6])
    ax.set_ylim([-6,6])
    plt.plot(x[0], y[0])
    plt.plot(x[1], y[1])
    plt.show()
    '''

    n_points = 100
    theta,phi = random_points_on_sphere(n_points)
    coords = random_points_on_circle(1000)
    new_coords = rotate_circle(coords, theta[0], phi[0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[0,:],coords[1,:],coords[2,:])
    ax.scatter(new_coords[0,:],new_coords[1,:],new_coords[2,:])

    x,y,z  = get_cartesian_coordinates(np.vstack((np.ones(n_points), theta, phi)))
    print(x,y,z)
    ax.plot([x[0],0],[y[0],0],[z[0],0])
    plt.show()

if __name__=='__main__':
    main()
