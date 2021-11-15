import numpy as np
import open3d as o3d
import shortuuid
import utility as utility
from time import time
import itertools
from copy import copy, deepcopy


class SphereTree(object):
    # Generates and Octree from a point cloud, then converts it to a sphere tree

    def __init__(self, file_path, min_voxel):
        t0 = time()
        # Generate the point cloud
        max_voxel_sa = (min_voxel ** 2) * np.sqrt(2)
        point_density = 15 / max_voxel_sa
        mesh = o3d.io.read_triangle_mesh(file_path)
        surface_area = mesh.get_surface_area()
        min_points = round(surface_area * point_density)
        pcd = mesh.sample_points_poisson_disk(min_points)
        self.geometry_points = np.array(pcd.points)
        t1 = time()

        # Get root node dimension
        bbox_x = max(self.geometry_points[:, 0]) - min(self.geometry_points[:, 0])
        bbox_y = max(self.geometry_points[:, 1]) - min(self.geometry_points[:, 1])
        bbox_z = max(self.geometry_points[:, 2]) - min(self.geometry_points[:, 2])
        self.root_length = max([bbox_x, bbox_y, bbox_z])
        self.root_radius = np.sqrt(3*self.root_length**2)/2

        # Allow "max_depth" to be minimum octree voxel size
        max_depth = round(np.log(self.root_length/min_voxel)/np.log(2))

        # Translate Part to bounding box center; for simplicity
        x_c = min(self.geometry_points[:, 0]) + bbox_x/2
        y_c = min(self.geometry_points[:, 1]) + bbox_y/2
        z_c = min(self.geometry_points[:, 2]) + bbox_z/2
        pcd.translate([-x_c, -y_c, -z_c])
        self.geometry_points = np.array(pcd.points)
        self.object_center = np.array([0, 0, 0])

        # Get xyz min/max
        # Credit to: https://codereview.stackexchange.com/questions/126521/python-octree-implementation
        xyzmin = np.array([-self.root_length/2]*3)
        xyzmax = np.array([self.root_length/2]*3)
        print(xyzmin, xyzmax)

        t2 = time()
        # Generate the octree & sphere tree
        nodes = utility.split(xyzmax, xyzmin)
        leafs = np.array(utility.generate_tree(self.geometry_points, nodes, max_depth=max_depth))
        self.leaf_radius, self.centers = utility.generate_spheres(leafs)
        t3 = time()

        print("Open3D Code Section: {}".format(t1-t0))
        print("My code: {}".format(t3-t2))

    def translate(self, vector, absolute=False):

        # Apply absolute translation to sphere centers
        if absolute:
            d = vector - self.object_center
            self.object_center = utility.translate(self.object_center, d)
            self.centers = utility.translate(self.centers, d)

        # Apply relative translation to sphere centers
        else:
            self.object_center = utility.translate(self.object_center, vector)
            self.centers = utility.translate(self.centers, vector)

    def rotate(self, vector, origin=False):

        # Apply point rotations about the origin
        if origin:
            self.object_center = utility.rotate(self.object_center, vector)
            self.centers = utility.rotate(self.centers, vector)

        # Apply rotation from "part" center (center of the root node bounding box)
        else:
            # Translate part to the origin
            self.centers = utility.translate(self.centers, -self.object_center)

            # Perform Rotation
            self.centers = utility.rotate(self.centers, vector)

            # Translate part back to where it started
            self.centers = utility.translate(self.centers, self.object_center)


class Collisions(object):

    def __init__(self, part_interval=2.5, envelope_interval=0):

        # Intervals
        self.part_interval = part_interval
        self.envelope_interval = envelope_interval

        # Setup Part Dictionary
        self.parts = {
        }

        # Combination Tracker
        self.collision_pairs = None

    def add_part(self, part):

        if not isinstance(part, SphereTree):
            raise Exception("Cannot at object of type {0} to Collision handler. Part must be of type {1}".format(
                type(part), SphereTree
            ))

        self.parts.update({shortuuid.ShortUUID().random(length=22):
                           part})

        self.update_collision_pairs()

    def add_parts(self, parts):

        for part in parts:
            self.add_part(part)

    def update_collision_pairs(self):

        ids = list(self.parts.keys())
        combos = list(itertools.combinations(ids, 2))
        self.collision_pairs = np.array(list(set(combos)))

    def check_collision(self, ids):

        # Gets total collisions between two parts given IDs
        r = self.parts[ids[0]].leaf_radius
        combos = np.array(list(itertools.product(*[self.parts[i].centers for i in ids])))

        # Check collisions
        collisions = utility.collision_check(combos, 2*r+self.part_interval)
        count = np.count_nonzero(collisions)

        return count

    def total_part_collisions(self):

        # Find Collision Pairs for Sphere Tree Collision Testing
        root_radius = list(self.parts.values())[0].root_radius
        point_pairs = np.array([(self.parts[pair[0]].object_center,
                                self.parts[pair[1]].object_center) for pair in self.collision_pairs])
        collisions = utility.collision_check(point_pairs, 2*root_radius+self.part_interval)

        # Get pairs of trees to perform deeper collision check
        tree_pairs = self.collision_pairs[np.where(collisions)[0]]
        total_collisions = 0
        for pair in tree_pairs:
            total_collisions += self.check_collision(pair)

        return total_collisions


if __name__ == "__main__":

    file = 'meshes/GenerativeBracket.stl'
    obj1 = SphereTree(file, 10)

    obj2 = SphereTree(file, 10)
    obj2.rotate([90, 0, 0])
    obj2.translate([5, 0, 60])

    obj3 = SphereTree(file, 10)
    obj3.rotate([0, 90, 0])
    obj3.translate([-5, 30, 0])

    col = Collisions()
    col.add_parts([obj1, obj2, obj3])
    collisions = col.total_part_collisions()
    print(collisions)

    utility.visualize_octree([obj1, obj2, obj3])
