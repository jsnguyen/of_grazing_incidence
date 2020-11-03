#!/usr/local/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import rebound

# may also be interpreted as random vectors that uniformly sample the surface of the sphere
def random_points_on_unit_sphere(n_points):

    if n_points == 1:
        u = np.random.uniform()
        v = np.random.uniform()

        theta = np.arccos(2*v-1) # inclination
        phi = 2*np.pi*u # azimuth

    else:
        theta = np.zeros(n_points)
        phi = np.zeros(n_points)

        for i in range(n_points):
            u = np.random.uniform()
            v = np.random.uniform()

            theta[i] = np.arccos(2*v-1) # inclination
            phi[i] = 2*np.pi*u # azimuth

    return theta, phi

def spherical_to_cartesian(coords):

    if len(coords.shape) == 1:
        rho = coords[0]
        theta = coords[1]
        phi = coords[2]

    else:
        rho = coords[0,:]
        theta = coords[1,:]
        phi = coords[2,:]

    x = rho*np.sin(theta)*np.cos(phi)
    y = rho*np.sin(theta)*np.sin(phi)
    z = rho*np.cos(theta)

    return np.array([x, y, z])

def random_points_on_circle(n_points):

    if n_points == 1:
        phi = 2*np.pi*np.random.uniform() # azimuth

        x = np.cos(phi)
        y = np.sin(phi)

    else:
        x = np.zeros(n_points)
        y = np.zeros(n_points)

        for i in range(n_points):
            phi = 2*np.pi*np.random.uniform() # azimuth

            x[i] = np.cos(phi)
            y[i] = np.sin(phi)

    return np.vstack((x,y))

def rotate_circle(coords, theta, phi):

    Ry = np.array([[np.cos(theta), 0 , np.sin(theta)],
                   [0,1,0],
                   [-np.sin(theta), 0 ,np.cos(theta)]])

    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])

    res = np.dot(Rz,np.dot(Ry,coords))

    return res

def random_tangent_line():
    theta,phi = random_points_on_unit_sphere(1)

    circle_coords = random_points_on_circle(1)
    new_circle_coords = np.vstack((circle_coords,0))
    new_coords = rotate_circle(new_circle_coords, theta, phi)

    translation = new_coords.reshape((1,3))[0]
    direction = spherical_to_cartesian(np.array([1,theta,phi]))

    return translation, direction

def draw_sphere(ax, radius):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

def get_line(translation, direction):
    line = np.vstack((translation-direction,translation,translation+direction))
    return line

def main():
    sim = rebound.Simulation()

    sim.G = 6.67430e-11

    solar_mass = 1.98847e30
    earth_mass = 5.9722e24
    au = 149597870700.0

    sim.add(m=1*solar_mass)
    sim.add(m=1*earth_mass, a=1*au)

    closest_approach = 50*1000*au # au
    radial_velocity = 83.1*1000 # m/s
    multiplier = 100

    delta_t = np.sqrt(multiplier*multiplier - 1) * closest_approach/radial_velocity
    half_distance = closest_approach * np.sqrt(multiplier*multiplier - 1)


    translation, direction = random_tangent_line()
    initial_position = (translation-direction) * closest_approach
    initial_velocity = radial_velocity * direction
    
    sim.add(m = 0.15*solar_mass,
            x = initial_position[0],
            y = initial_position[1],
            z = initial_position[2],
            vx = initial_velocity[0],
            vy = initial_velocity[1],
            vz = initial_velocity[2])

    sim.move_to_com()

    ps = sim.particles

    n_iter = 100000
    times = np.linspace(0,delta_t*2, n_iter)

    x = np.zeros((3,n_iter))
    y = np.zeros((3,n_iter))
    z = np.zeros((3,n_iter))

    for i,time in enumerate(times):
        print(i)
        sim.integrate(time)
        x[0][i] = ps[0].x
        y[0][i] = ps[0].y
        z[0][i] = ps[0].z
        x[1][i] = ps[1].x
        y[1][i] = ps[1].y
        z[1][i] = ps[1].z
        x[2][i] = ps[2].x
        y[2][i] = ps[2].y
        z[2][i] = ps[2].z

    sim.status()

    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot(111)
    #ax.set_xlim(-closest_approach,closest_approach)
    #ax.set_ylim(-closest_approach,closest_approach)
    plt.plot(x[0], y[0])
    plt.plot(x[1], y[1])
    plt.plot(x[2], y[2])
    plt.show()

    plt.show()

if __name__=='__main__':
    main()
