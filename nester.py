import geometry as geo
from copy import copy, deepcopy


class Nester(object):

    def __init__(self, scene, custom_functions=None):

        self.scene = scene
        self.custom_fuctions = {}

        # Create a "reference part" and translate it to the middle of the build envelope
        self.reference_part = list(scene.parts.values())[0]
        self.reference_part.translate(self.scene.build_envelope.node[1]/2)

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

    def advance_state(self):

        # Add a part to the scene
        new_part = deepcopy(self.reference_part)

        #



if __name__ == "__main__":

    # Create a part, envelope, and scene
    part = geo.SphereTree('meshes/GenerativeBracket.stl', 10)
    scene = geo.Scene()
    envelope = geo.Envelope(380, 284, 380)

    # Add Elements to the Scene
    scene.add_part(part)
    scene.add_envelope(envelope)

    # Create the Nester
    nester = Nester(scene)
    nester.advance_state()

