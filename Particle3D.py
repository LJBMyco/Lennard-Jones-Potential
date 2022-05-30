import numpy as np
import random

class Particle3D(object):

    def __init__(self, mass, label, position, velocity):

        self.mass = mass
        self.label = label
        self.position = position
        self.velocity = velocity

    def __str__(self):

        return f'{str(self.label)} {str(self.position[0])} {self.position[1]} {self.position[2]}'

    def kinetic_energy(self):

        return 0.5*self.mass* (np.linalg.norm(self.velocity)**2.0)

    def velocity_update(self, dt, force):

        self.velocity += dt*(force/self.mass)

    def position_update(self, dt, force):

        self.position += dt*self.velocity + 0.5*(dt**2.0)*(force/self.mass)

    @staticmethod
    def generate_particle(index):

        mass = 1.0
        label = str(index+1)
        postion = np.zeros(3)
        velocity = np.zeros(3)

        return Particle3D(mass, label, postion, velocity)

    def vector_separation(pos1, pos2):

        return pos1-pos2

    def pbc(vector, box_size):

        return np.mod(vector, box_size)

    def mic(vector, box_size):

        mic_vector = np.zeros(len(vector))
        mic_vector = np.mod((vector + box_size/2.0), box_size) - (box_size/2.0)

        return mic_vector
