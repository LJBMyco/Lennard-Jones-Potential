import numpy as np
import sys
import math as m
from Particle3D import Particle3D
from MDUtilities import SetInitialPositions
from MDUtilities import SetInitialVelocities
import typing


########## Separations ##########

"""Calculate the separation between all particles"""


def create_separation_array(particles: list, box_size: np.array, particle_number: int) -> np.array:

    separation_array = np.zeros((particle_number, particle_number, 3))
    for i in range(particle_number):
        for j in range(0, i):
            # Zero alon diagonals
            if i != j:
                separation = Particle3D.vector_separation(
                    particles[i].position, particles[j].position)
                # Apply MIC
                mic_separation = Particle3D.mic(separation, box_size)
                #S_ij = -S_ji
                separation_array[i][j] = mic_separation
                separation_array[j][i] = -1.0*mic_separation

    return separation_array

########## Forces  ##########


"""Calculate force from LJ potential"""


def calculate_force(separation: float, direction_vector: np.array) -> np.array:

    return 48.0*((separation**-14.0)-0.5*(separation**-8.0)) * direction_vector


"""Calculate total force on each particle"""


def create_force_array(separation_array: np.array, particles: list, cut_distance: float, particle_number: int) -> np.array:

    # Calculate pairwise forces
    force_array = np.zeros((particle_number, particle_number, 3))
    for i in range(particle_number):
        for j in range(0, i):
            if i != j:
                separation = separation_array[i][j]
                separation_mag = np.linalg.norm(separation)
                # Account for cut off distance
                if separation_mag <= cut_distance:
                    force = calculate_force(separation_mag, separation)
                    # Use NIII
                    force_array[i][j] = force
                    force_array[j][i] = -1.0*force

    # Total force on each particle sum of all pairwise forces
    total_force_array = np.zeros((particle_number, 3))
    for i in range(particle_number):
        for j in range(particle_number):
            total_force_array[i] += force_array[i][j]

    return total_force_array

########## Energy ##########


"""Calculate potential energy between two particles from LJ Potential"""


def calculate_potential(separation: float) -> float:

    return 4.0*((separation**-12.0)-(separation**-6.0))


"""Calculate total potential energy of the simulation"""


def total_potential(separation_array: np.array, particle_number: int, cut_distance: float) -> np.array:

    # Calculate pairwise potential
    potential_array = np.zeros((particle_number, particle_number))

    for i in range(particle_number):
        for j in range(0, i):
            # Zero along diagonals
            if i != j:
                separation = separation_array[i][j]
                separation_mag = np.linalg.norm(separation)
                if separation_mag <= cut_distance:
                    potential = calculate_potential(separation_mag)
                    # Only need upper matrix to avoid daouble counting
                    potential_array[i][j] = potential

    return np.sum(potential_array)


"""Calculate the toral kinetice energy of the simulation"""


def total_kinetic(particles: list, particle_number: int) -> float:

    total_ke = 0.0
    for particle in particles:
        total_ke += particle.kinetic_energy()

    return total_ke

########## Observables ##########


def mean_square_displacement(initial_positions, particles, particle_number, box_size):

    msd = 0.0
    for i, particle in enumerate(particles):
        travel_vector = Particle3D.vector_separation(
            particle.position, initial_positions[i])
        tv_mic = Particle3D.mic(travel_vector, box_size)
        msd += np.linalg.norm(tv_mic)**2.0

    return msd/particle_number

########## Outputs ##########


"""Write .xyz file"""


def write_traj(particles: list, particle_number: int, point: int, out_file_handle: str) -> None:

    out_file_handle.write(str(particle_number) + '\n')
    out_file_handle.write(point + '\n')
    for particle in particles:
        out_file_handle.write(str(particle) + '\n')

########## Velocity Verlet update ##########


"""Use velocity verlet time integration method to update positions and velocities"""


def velocity_verlet(particles: list, dt: float, box_size: np.array, particle_number: int, force_array: np.array, cut_distance: float) -> typing.Tuple[list, np.array, np.array]:

    # Update positions
    for i, particle in enumerate(particles):
        particle.position_update(dt, force_array[i])
        # Use pbc
        particle.position = Particle3D.pbc(particle.position, box_size)

    # Update separations
    separation_array = create_separation_array(
        particles, box_size, particle_number)

    # Update forces
    new_force_array = create_force_array(
        separation_array, particles, cut_distance, particle_number)

    # Update velocities
    for i, particle in enumerate(particles):
        # Use mean of old force and new force
        vel_update = 0.5*(force_array[i]+new_force_array[i])
        particle.velocity_update(dt, vel_update)

    # Copy new forces
    force_array = new_force_array

    return particles, force_array, separation_array


def main():

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

    # Open out file
    out_file_handle = open('traj.xyz', 'w')

    # Initialise simulaiton
    time = 0.0

    particles = [Particle3D.generate_particle(
        i) for i in range(particle_number)]

    box_size = SetInitialPositions(density, particles)
    SetInitialVelocities(temp, particles)

    separation_array = create_separation_array(
        particles, box_size, particle_number)
    initial_positions = np.copy(separation_array)
    force_array = create_force_array(
        separation_array, particles, cut_distance, particle_number)

    total_ke = total_kinetic(particles, particle_number)
    total_pe = total_potential(separation_array, particle_number, cut_distance)
    total_energy = total_ke + total_pe
    msd = mean_square_displacement(
        initial_positions, particles, particle_number, box_size)

    # Create output lists
    time_list = []
    ke_list = []
    pe_list = []
    total_energy_list = []
    msd_list = []

    # Output first step
    inital_point = 'Point = 1'
    write_traj(particles, particle_number, inital_point, out_file_handle)
    time_list.append(time)
    ke_list.append(total_ke)
    pe_list.append(total_pe)
    total_energy_list.append(total_energy)
    msd_list.append(msd)

    # Begin simulation
    for n in range(step_number):

        # Velocity verlet update
        particles, force_array, separation_array = velocity_verlet(
            particles, dt, box_size, particle_number, force_array, cut_distance)

        # Increment time
        time += dt

        # Calculate observables
        total_ke = total_kinetic(particles, particle_number)
        total_pe = total_potential(
            separation_array, particle_number, cut_distance)
        total_energy = total_ke + total_pe
        msd = mean_square_displacement(
            initial_positions, particles, particle_number, box_size)

        # Output data
        point = "Point = " + str(n + 2)
        write_traj(particles, particle_number, point, out_file_handle)
        time_list.append(time)
        ke_list.append(total_ke)
        pe_list.append(total_pe)
        total_energy_list.append(total_energy)
        msd_list.append(msd)

        print(f'{n}/{step_number}', end='\r')

    np.save('time.npy', time_list)
    np.save('kinetic_energy.npy', ke_list)
    np.save('potential_energy.npy', pe_list)
    np.save('total_energy.npy', total_energy_list)
    np.save('msd.npy', msd_list)


if __name__ == '__main__':
    main()
