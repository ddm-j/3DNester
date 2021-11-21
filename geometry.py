import numpy as np
import open3d as o3d
import shortuuid
import utility
import itertools
import warnings
import cmapy
import random
import utility_cy as uc
from copy import copy, deepcopy


class SphereTree(object):
    # Generates and Octree from a point cloud, then converts it to a sphere tree

    def __init__(self, file_path, min_voxel):

        # Generate the point cloud, prepare for affine transforms
        max_voxel_sa = (min_voxel ** 2) * np.sqrt(2)
        point_density = 15 / max_voxel_sa
        mesh = o3d.io.read_triangle_mesh(file_path)
        surface_area = mesh.get_surface_area()
        min_points = round(surface_area * point_density)
        pcd = mesh.sample_points_poisson_disk(min_points)
        ones = np.ones((1, len(pcd.points))).T
        self.geometry_points = np.concatenate((np.array(pcd.points), ones), axis=1)

        # Get root node dimension
        bbox_x = max(self.geometry_points[:, 0]) - min(self.geometry_points[:, 0])
        bbox_y = max(self.geometry_points[:, 1]) - min(self.geometry_points[:, 1])
        bbox_z = max(self.geometry_points[:, 2]) - min(self.geometry_points[:, 2])
        self.bbox_array = 0.5*np.array([
            [-bbox_x, -bbox_y, -bbox_z, 1],
            [bbox_x, -bbox_y, -bbox_z, 1],
            [bbox_x, bbox_y, -bbox_z, 1],
            [-bbox_x, bbox_y, -bbox_z, 1],
            [-bbox_x, -bbox_y, bbox_z, 1],
            [bbox_x, -bbox_y, bbox_z, 1],
            [bbox_x, bbox_y, bbox_z, 1],
            [-bbox_x, bbox_y, bbox_z, 1]
        ])
        self.bbox = [bbox_x, bbox_y, bbox_z, 1]
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
        self.center = np.array([0, 0, 0, 1])

        # Get xyz min/max
        # Credit to: https://codereview.stackexchange.com/questions/126521/python-octree-implementation
        xyzmin = np.array([-self.root_length/2]*3)
        xyzmax = np.array([self.root_length/2]*3)

        # Generate the octree & sphere tree, format sphere points for affine transforms
        nodes = utility.split(xyzmax, xyzmin)
        leafs = np.array(utility.generate_tree(self.geometry_points, nodes, max_depth=max_depth))
        self.leaf_radius, centers = utility.generate_spheres(leafs)
        ones = np.ones((1, len(centers))).T
        self.points = np.concatenate((centers, ones), axis=1)

        # Generate Basic Translation, Rotation, and history matrices
        self.position_matrix = np.eye(4)
        self.trans_matrix = np.eye(4)
        self.rot_matrix = np.zeros((4,4))
        self.rot_matrix[3, 3] = 1

    def apply_transforms(self):

        # Outputs part center and points after transformations are applied.
        # If inplace is used, the transforms will be applied to the part. Rotations section will be reset (identity)

        center = utility.transform(self.center, self.position_matrix, rows=True)
        points = utility.transform(self.points, self.position_matrix, rows=True)

        return center, points

    def translate(self, vector, absolute=False):

        # Convert vector to list
        vector = list(vector)

        # Format vector for affine
        if len(vector) == 3:
            vector.append(1)
        matrix = copy(self.trans_matrix)

        # Apply absolute translation to position matrix
        if absolute:
            d = vector - self.center
            matrix[:, 3] = d

        # Apply relative translation to position matrix
        else:
            matrix[:, 3] = vector

        # Perform Affine Transform
        self.position_matrix = utility.transform(self.position_matrix, matrix)

    def rotate(self, vector, matrix=False):
        # Preprocess arguments
        if matrix:
            # "Vector" argument is actually a rotation matrix
            mat = vector
        else:
            # Get Rotation Matrix
            mat = utility.rotation_matrix(vector)

        # Get the stock affine rotation matrix
        matrix = copy(self.rot_matrix)
        matrix[0:3, 0:3] = mat

        # Perform Rotation about part origin
        self.position_matrix = utility.transform(self.position_matrix, matrix)


        # Update bounding box array
        self.bbox_array = utility.transform(self.bbox_array, matrix, rows=True)
        self.bbox = [
            max(self.bbox_array[:, 0]) - min(self.bbox_array[:, 0]),
            max(self.bbox_array[:, 1]) - min(self.bbox_array[:, 1]),
            max(self.bbox_array[:, 2]) - min(self.bbox_array[:, 2])
        ]

    def visualize(self, option='tree', axes=True, cmap='gist_rainbow', discrete=True):

        # Visualize the current geometry

        # Apply transforms to the geometry
        _, points = self.apply_transforms()

        # Function Variables
        geometries = []

        # Generate Coordinate Axes
        if axes:
            geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=75))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
        if discrete:
            rgb_colors = np.array([cmapy.color(cmap, random.randrange(0, 256, 10), rgb_order=True)
                                   for i in range(len(points))])
        else:
            rgb_colors = np.array([cmapy.color(cmap, random.randrange(0, 256), rgb_order=True)
                                   for i in range(len(points))])

        pcd.colors = o3d.utility.Vector3dVector(rgb_colors.astype(np.float) / 255.0)
        d = self.leaf_radius * 2
        size = np.sqrt((d ** 2) / 2)

        if option == 'tree':
            geometries.append(o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, size))
        elif option == 'pcd':
            geometries.append(pcd)

        # Start the visualization
        o3d.visualization.draw_geometries(geometries)


class Scene(object):

    def __init__(self, part_interval=2.5, envelope_interval=0):

        # Intervals
        self.part_interval = part_interval
        self.envelope_interval = envelope_interval

        # Setup Scene Data Structures
        # Center Point List (tracks origin points of all objects in the scene)
        self.cp_arr = []
        # Rotation Matrix List (tracks
        self.rot_arr = []

        # Combination Tracker
        self.collision_pairs = None

        # Envelope
        self.build_envelope = None

    def add_part(self, part, in_loop=False):

        if not isinstance(part, SphereTree):
            raise Exception("Cannot at object of type {0} to Collision handler. Part must be of type {1}".format(
                type(part), SphereTree
            ))

        self.parts.update({shortuuid.ShortUUID().random(length=22):
                           part})

        # Update the collision pairs (if this is a single function call, not in a loop)
        if not in_loop:
            self.update_collision_pairs()

    def remove_part(self, id, in_loop=False):

        # Remove a part given it's ID
        self.parts.pop(id)

        # Update the collision pairs (if this is a single function call, not in a loop)
        if not in_loop:
            self.update_collision_pairs()

    def remove_parts(self, ids):

        # Remove parts given their IDs
        for id in ids:
            self.remove_part(id, in_loop=True)

        # Update collision pairs
        self.update_collision_pairs()

    def add_parts(self, parts):

        for part in parts:
            self.add_part(part)

        # Update collision pairs
        self.update_collision_pairs()

    def add_envelope(self, envelope):

        self.build_envelope = envelope

    def update_collision_pairs(self):

        ids = list(self.parts.keys())
        combos = list(itertools.combinations(ids, 2))
        self.collision_pairs = np.array(list(set(combos)))

    def check_collision(self, ids, method='cy'):

        # Gets total collisions between two parts given IDs
        r = self.parts[ids[0]].leaf_radius

        if method == None:
            combos = np.array(list(itertools.product(*[self.parts[i].centers for i in ids])))
        elif method == 'cy':
            combos = uc.collision_pairs(self.parts[ids[0]].centers, self.parts[ids[1]].centers)
        elif method == 'new':
            combos = utility.cartesian(*[self.parts[i].centers for i in ids])

        # Check collisions
        collisions = utility.sphere_collision_check(combos, 2 * r + self.part_interval)

        count = np.count_nonzero(collisions)

        return count

    def total_part_collisions(self):
        total_collisions = 0
        if len(self.parts) > 1:

            # Find Collision Pairs for Sphere Tree Collision Testing
            root_radius = list(self.parts.values())[0].root_radius
            point_pairs = np.array([(self.parts[pair[0]].object_center,
                                    self.parts[pair[1]].object_center) for pair in self.collision_pairs])
            collisions = utility.sphere_collision_check(point_pairs, 2 * root_radius + self.part_interval)

            # Perform deeper collision check (computationally expensive)
            tree_pairs = self.collision_pairs[np.where(collisions)[0]]

            # Process "tree_pairs" into an (m, 2, n, 3) array
            point_arr = np.array([
                [self.parts[p[0]].centers, self.parts[p[1]].centers] for p in tree_pairs
            ])
            uc.all_collision_pairs(point_arr)

            for pair in tree_pairs:
                total_collisions += self.check_collision(pair)

        return total_collisions

    def total_envelope_collisions(self):

        if not isinstance(self.build_envelope, Envelope):
            self.add_envelope(Envelope(380, 284, 380))
            warnings.warn("No build volume added (class method add_envelope). Defaulting to HP MJF 4200 envelope.")

        # Get Parts whose bounding sphere are not completely inside the envelope
        ids = list(self.parts.keys())
        vectors = np.array([self.parts[i].object_center for i in ids])
        root_radius = list(self.parts.values())[0].root_radius
        collisions = utility.rect_collision_check(vectors, self.build_envelope.node, root_radius)

        # Get pruned list for sphere tree collision testing
        tree_pairs = np.array(ids)[np.where(collisions)[0]]
        if len(tree_pairs) != 0:
            print('{0} Parts may be out of bounds. Performing deeper check.'.format(len(tree_pairs)))
        total_collisions = 0
        for id in tree_pairs:
            vectors = self.parts[id].centers
            collisions = utility.rect_collision_check(vectors, self.build_envelope.node, self.parts[id].leaf_radius+self.envelope_interval)
            total_collisions += np.count_nonzero(collisions)

        return total_collisions

    def total_collisions(self):

        return self.total_part_collisions() + self.total_envelope_collisions()

    def visualize(self, option='tree', axes=True, cmap='gist_rainbow', discrete=False):
        # Function Variables
        geometries = []

        # Generate Coordinate Axes
        if axes:
            geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=75))

        # Generate  Part Objects
        for part in self.parts.values():
            # Create octree point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(part.centers)
            if discrete:
                rgb_colors = np.array([cmapy.color(cmap, random.randrange(0, 256, 10), rgb_order=True)
                                       for i in range(len(part.centers))])
            else:
                rgb_colors = np.array([cmapy.color(cmap, random.randrange(0, 256), rgb_order=True)
                                       for i in range(len(part.centers))])
            pcd.colors = o3d.utility.Vector3dVector(rgb_colors.astype(np.float) / 255.0)
            d = part.leaf_radius * 2
            size = np.sqrt((d ** 2) / 2)

            if option == 'tree':
                geometries.append(o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, size))
            elif option == 'pcd':
                geometries.append(pcd)

        # Create the build envelope
        if self.build_envelope:
            x, y, z = self.build_envelope.xyzmax
            # Points in right-hand rule order, bottom to top from origin
            points = np.array([
                [0.0, 0.0, 0.0],  # 0
                [x, 0.0, 0.0],  # 1
                [x, y, 0.0],  # 2
                [0, y, 0.0],  # 3
                [0.0, 0.0, z],  # 4
                [x, 0.0, z],  # 5
                [x, y, z],  # 6
                [0, y, z]  # 7
            ])
            lines = np.array([
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [3, 7]
            ])

            # Create the line-set
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(points)
            line_set.lines = o3d.utility.Vector2iVector(lines)

            # Append to geometries
            geometries.append(line_set)

        # Start the visualization
        o3d.visualization.draw_geometries(geometries)


class Envelope(object):

    def __init__(self, x, y, z, units='mm'):

        # Typechecking
        if not all(isinstance(i, (float, int)) for i in [x, y, z]):
            raise Exception('Build envelope size must be of int or float.')
        if units not in ['mm', 'cm', 'm', 'in', 'ft']:
            raise Exception('Unit selection can only be: mm, cm, m, in, ft (default mm)')

        # Unit Conversions
        if units == 'in':
            factor = 25.4
        elif units == 'cm':
            factor = 10.0
        elif units == 'm':
            factor = 1000
        elif units == 'ft':
            factor = 304.8
        else:
            factor = 1.0

        # BBox Vectors
        self.xyzmin = np.array([0.0, 0.0, 0.0])
        self.xyzmax = np.array([factor*x, factor*y, factor*z])
        self.node = np.array([self.xyzmin, self.xyzmax])

        # Build Volume
        self.build_volume = np.prod(self.xyzmax)

    def calculate_packing_density(self, part_volume, n_parts):

        # Packing Density Evalutation Function

        return (part_volume*n_parts)/self.build_volume


if __name__ == "__main__":

    file = 'meshes/GenerativeBracket.stl'
    obj1 = SphereTree(file, 10)
    print('Translating Part')
    obj1.translate([380.0/2.5, 284.0/4+30, 60+20])
    obj1.visualize()
    print('Current Position Matrix')
    print(obj1.position_matrix)
    print('Rotating Part')
    obj1.rotate([90, 0, 0])
    obj1.visualize()
    print('Current Position Matrix')
    print(obj1.position_matrix)
    print('Center Before Transform')
    print(obj1.center)
    obj1.apply_transforms()
    obj1.visualize()
    print('Center after transform')
    print(obj1.center)
    print('Reset Position Matrix (rotations)')
    print(obj1.position_matrix)


'''
    obj2 = SphereTree(file, 10)
    obj2.rotate([90, 0, 0])
    obj2.rotate([45, 90, 0])
    obj2.translate([380.0/2, 284.0/2, 380.0/2])

    obj3 = SphereTree(file, 10)
    obj3.rotate(obj2.total_rotation_matrix, matrix=True)
    obj3.translate([380.0/3.2, 284.0/1.5, 380.0/2+5])

    scene = Scene(part_interval=1.5)
    scene.add_parts([obj1, obj2, obj3])
    scene.add_envelope(Envelope(380, 284, 380))
    collisions = scene.total_collisions()
    print(collisions)

    scene.visualize(cmap='Greys')
    
'''
