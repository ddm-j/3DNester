import geometry as geo
from copy import copy, deepcopy
import random
import numpy as np
import pandas as pd
import utility


class Nester(object):

    def __init__(self, scene, custom_functions=None, limit_rotation=True):

        # Initialize data structures
        self.scene = scene
        self.custom_fuctions = {}
        self.limit_rotation = limit_rotation

        # Create a "reference part" and translate the "original" part to the center of the BV
        self.reference_part = deepcopy(list(scene.parts.values())[0])
        list(scene.parts.values())[0].translate(self.scene.build_envelope.node[1]/2)

        # Create distances for translate moves (mm)
        distance = np.linspace(0.5, 30, 15)

        # Create move-set, mx6 array. Move type, arguments, n_prev_attempts, delta C, quality, probability
        self.p_min = 0.02
        move_set = [
            ['rotate', None, 0, 0, 0, self.p_min],
            ['swap', None, 0, 0, 0, self.p_min],
            ['add', None, 0, 0, 0, 0.8],
            ['remove', None, 0, 0, 0, 0.005]
        ]
        for d in distance:
            move_set.append(['translate', d, 0, 0, 0, self.p_min])
        self.move_set = pd.DataFrame(move_set, columns=['move', 'params', 'count', 'dC', 'quality', 'prob'])

    def evaluate_state(self, collision_weight=1.0):

        # Evaluate the "fitness" of the current scene configuration
        # Returns collision_weight*collisions + sum(custom_functions)

        # Calculate No. Collisions in the scene
        collisions = self.scene.total_collisions()

        # Calculate the sum of user provided optimization functions
        custom = 0
        if len(self.custom_fuctions) > 0:
            for key in self.custom_fuctions.keys():
                func = self.custom_fuctions[key][1]
                weight = self.custom_fuctions[key][0]
                res = func(scene)
                custom += res*weight

        return collision_weight*collisions + custom

    def perturb(self, move):
        # Perturb the current state using "move" (selection from move-set) and return it's fitness
        # All perturbations require a randomly selected part
        part = random.choices(list(self.scene.parts.keys()), k=1)[0]
        move_type = move['move'].values

        #print(self.scene.parts[part].object_center)

        if move_type == 'translate':
            # Generate a random vector and translate our part (relatively)
            print('Translating')
            vec = utility.make_rand_vector(3, move.params)
            self.scene.parts[part].translate(vec)
        elif move_type == 'rotate':
            print('Rotating')
            # Generate random rotation vector
            if self.limit_rotation:
                vec = [random.randint(0, 3)*90 for n in range(3)]
            else:
                vec = [random.uniform(0, 360) for n in range(3)]
            self.scene.parts[part].rotate(vec)
        elif move_type == 'remove':
            if len(self.scene.parts) > 1:
                print('Removing')
                # Remove the part from the scene
                self.scene.remove_part(part)
        elif move_type == 'add':
            # Create a new part instance and add it to the scene
            print('Adding')
            new = deepcopy(self.reference_part)

            # Randomly Rotate the part
            if self.limit_rotation:
                rot = [random.randint(0, 3)*90 for n in range(3)]
            else:
                rot = [random.uniform(0, 360) for n in range(3)]
            new.rotate(rot)

            # Position part next to an existing part
            vec = copy(self.scene.parts[part].object_center)
            ind = random.randint(0, 2)
            d = (new.bbox[ind]+self.scene.parts[part].bbox[ind])*0.5+10
            vec[ind] += random.choices([-1, 1], k=1)[0]*d

            # Translate the part & add to scene
            new.translate(vec)

            self.scene.add_part(new)
        elif move_type == 'swap':
            if len(self.scene.parts) > 2:
                print('Swapping')
                opt = list(self.scene.parts.keys())
                opt.remove(part)
                part2 = random.choices(opt, k=1)[0]

                # Get Current Part Centers
                c1 = copy(self.scene.parts[part].object_center)
                c2 = copy(self.scene.parts[part2].object_center)

                # Swap positions of these parts
                self.scene.parts[part].translate(c2, absolute=True)
                self.scene.parts[part2].translate(c1, absolute=True)

        fitness = self.evaluate_state()

        print(fitness)

        return fitness

    def anneal(self):

        # Pre-process the state space. Calculate random-walk variance of the state space. Initialize Temperature based
        # on the variance of the random walk.
        t = 1

        # Initialize the tracking of change in the objective function per move in the move-set
        dC = []

        # While t > 0
        while t > 0:
            # Reset maximum function values & normalizing factors @ this new timestep

            search = True
            j = 0
            while j < 50:
                # Initialize counter for move tests at this temperature
                # Generate a new state & test fitness
                move = self.move_set.sample(n=1, weights=self.move_set['prob'])
                f_new = self.perturb(move)
                #self.scene.visualize(option='tree',cmap='Greys')
                j += 1
                #break

            break


if __name__ == "__main__":

    # Create a part, envelope, and scene
    part = geo.SphereTree('meshes/GenerativeBracket.stl', 10)
    scene = geo.Scene(part_interval=0, envelope_interval=0)
    envelope = geo.Envelope(380, 284, 380)

    # Add Elements to the Scene
    scene.add_part(part)
    scene.add_envelope(envelope)

    # Create the Nester
    nester = Nester(scene)
    nester.anneal()
    scene.visualize()
