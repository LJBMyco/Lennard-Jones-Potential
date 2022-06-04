import numpy as np
import sys
import math as m
from Particle3D import Particle3D
from MDUtilities import SetInitialPositions
from MDUtilities import SetInitialVelocities
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import pandas as pd

from datetime import datetime


def calculate_separations(ri, rj, box_size):

    M = ri.shape[0]
    N = rj.shape[0]

    # get x, y of r_i
    r_i_x = ri[:, 0].reshape((M, 1))
    r_i_y = ri[:, 1].reshape((M, 1))
    r_i_z = ri[:, 2].reshape((M, 1))

    # get x, y of r_j
    r_j_x = rj[:, 0].reshape((N, 1))
    r_j_y = rj[:, 1].reshape((N, 1))
    r_j_z = rj[:, 2].reshape((N, 1))

    # get r_i - r_j
    dx = Particle3D.mic(r_i_x - r_j_x.T, box_size[0])
    dy = Particle3D.mic(r_i_y - r_j_y.T, box_size[1])
    dz = Particle3D.mic(r_i_z - r_j_z.T, box_size[2])

    mag = np.sqrt(dx**2.0 + dy**2.0 + dz**2.0)

    return dx, dy, dz, mag


def calculate_force(sep):
    force = 48.0*((sep**-14.0)-0.5*(sep**-8.0))
    return force


def create_froce_array(pos, dx, dy, dz, mag, cut_ditstance):

    N = pos.shape[0]

    fx = calculate_force(mag)*dx
    fy = calculate_force(mag)*dy
    fz = calculate_force(mag)*dz

    fx[np.isnan(fx)] = 0.0
    fy[np.isnan(fy)] = 0.0
    fz[np.isnan(fz)] = 0.0

    fx[mag >= cut_ditstance] = 0.0
    fy[mag >= cut_ditstance] = 0.0
    fz[mag >= cut_ditstance] = 0.0

    fx = np.sum(fx, 1).reshape((N, 1))
    fy = np.sum(fy, 1).reshape((N, 1))
    fz = np.sum(fz, 1).reshape((N, 1))

    return fx, fy, fz

def calculate_potential(sep):
    return 4.0*((sep**-12.0)-(sep**-6.0))


def total_potential(mag, cut_ditstance):

    potential = calculate_potential(mag)
    potential[(mag==0.0) | (mag>=cut_ditstance)] = 0.0 
    return np.sum(potential)

def total_kinetic(particles):
    return np.sum([particle.kinetic_energy() for particle in particles])


def write_traj(particles: list, particle_number: int, point: int, out_file_handle: str) -> None:

    out_file_handle.write(str(particle_number) + '\n')
    out_file_handle.write(point + '\n')
    for particle in particles:
        out_file_handle.write(str(particle) + '\n')


def main():

    start_time = datetime.now()
    print(f'{start_time}: Running simulation')

    #Load in data
    in_file_handle = open('param.input.txt')
    line = in_file_handle.readline()
    tokens = line.split(',')
    particle_number = int(tokens[0])
    step_number = int(tokens[1])
    dt = float(tokens[2])
    temp = float(tokens[3])
    density = float(tokens[4])
    cut_distance = float(tokens[5])

    #Open out file
    out_file_handle = open('Output/traj.xyz', 'w')

    time = 0.0

    particles = [Particle3D.generate_particle(
        i) for i in range(particle_number)]

    box_size = SetInitialPositions(density, particles)
    SetInitialVelocities(temp, particles)

    positions = np.array([particle.position for particle in particles])
    sep_x, sep_y, sep_z, mag = calculate_separations(
        positions, positions, box_size)


    fx, fy, fz = create_froce_array(
        positions, sep_x, sep_y, sep_z, mag, cut_distance)

    total_ke = total_kinetic(particles)
    total_pe = total_potential(mag, cut_distance)
    total_energy = total_ke + total_pe

    # Create output lists
    time_list = []
    ke_list = []
    pe_list = []
    total_energy_list = []

    inital_point = 'Point = 1'
    write_traj(particles, particle_number, inital_point, out_file_handle)
    time_list.append(time)
    ke_list.append(total_ke)
    pe_list.append(total_pe)
    total_energy_list.append(total_energy)
    
    force = np.hstack((fx, fy, fz))
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.set(xlim=(0, 5), ylim=(0, 5), zlim=(0, 5))
    # ims = []

    data_load_time = datetime.now()
    print(f'{data_load_time}: Data and initial conditions loaded')

    with tqdm(total=step_number) as pbar:
        for n in range(step_number):

            for i, particle in enumerate(particles):
                particle.position_update(dt, force[i])
                # Use pbc
                particle.position = Particle3D.pbc(particle.position, box_size)

            positions = np.array([particle.position for particle in particles])

            sep_x, sep_y, sep_z, mag = calculate_separations(
                positions, positions, box_size)
            fx, fy, fz = create_froce_array(
                positions, sep_x, sep_y, sep_z, mag, cut_distance)

            new_force = np.hstack((fx, fy, fz))

            for i, particle in enumerate(particles):
                # Use mean of old force and new force
                vel_update = 0.5*(force[i]+new_force[i])
                particle.velocity_update(dt, vel_update)

            force = new_force

            time += dt 

            total_ke = total_kinetic(particles)
            total_pe = total_potential(mag, cut_distance)
            total_energy = total_ke + total_pe

            point = "Point = " + str(n + 2)
            write_traj(particles, particle_number, point, out_file_handle)
            time_list.append(time)
            ke_list.append(total_ke)
            pe_list.append(total_pe)
            total_energy_list.append(total_energy)

            # im = ax.scatter(positions[:,0],positions[:,1],positions[:,2], s=10, alpha=0.75)

            # ims.append([im])

            pbar.update(1)
            pbar.set_description(f'{datetime.now()}: Running')

            if n == step_number-1:
                pbar.set_description(f'{datetime.now()}: Finished')

    # ani = animation.ArtistAnimation(fig, ims)
    # ani.save('star_OG.gif')

    simulation_end_time = datetime.now()
    print(f'{simulation_end_time}: Simulation completed in {simulation_end_time-start_time}')

    handles = ['Time', 'Kinetic Energy', 'Potential Energy',
               'Total Energy']
    data = pd.DataFrame(data=np.array(
        [time_list, ke_list, pe_list, total_energy_list]).T, columns=handles)
    data.to_excel('Output/data.xlsx',
                  'Sheet1', index=None)

    print(f'{datetime.now()}: Data saved')

if __name__ == '__main__':
    main()
