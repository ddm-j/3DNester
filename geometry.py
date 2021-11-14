import numpy as np
import open3d as o3d


def rotation_matrix(x, y, z, degrees=True):
    # Vector Utility Function
    # Takes rotations on x, y, z axes and outputs a (3, 3) rotation matrix

    # Convert degrees to radians if needed
    if degrees:
        x, y, z = map(np.deg2rad, [x, y, z])

    # Switch from x, y, z rotation vector to yaw, pitch, roll
    # More info https://en.wikipedia.org/wiki/Rotation_matrix
    a, b, g = z, y, x

    # Construct rows of the rotation matrix
    r1 = [np.cos(a)*np.cos(b),
          np.cos(a)*np.sin(b)*np.sin(g)-np.sin(a)*np.cos(g),
          np.cos(a)*np.sin(b)*np.cos(g)+np.sin(a)*np.sin(g)]

    r2 = [np.sin(a)*np.cos(b),
          np.sin(a)*np.sin(b)*np.sin(g)+np.cos(a)*np.cos(g),
          np.sin(a)*np.sin(b)*np.cos(g)-np.cos(a)*np.sin(g)]

    r3 = [-np.sin(b),
          np.cos(b)*np.sin(g),
          np.cos(b)*np.cos(g)]

    # Assemble the rotation matrix
    mat = np.array([r1, r2, r3])

    # Set "almost zero" values to 0
    mat[np.abs(mat) < np.finfo(np.float).eps] = 0

    return mat


class SphereTree(object):
    # Generates and Octree from a point cloud, then converts it to a sphere tree

    def __init__(self, file_path, min_voxel):

        # Generate the point cloud
        max_voxel_sa = (min_voxel ** 2) * np.sqrt(2)
        point_density = 15 / max_voxel_sa
        mesh = o3d.io.read_triangle_mesh(file_path)
        surface_area = mesh.get_surface_area()
        min_points = round(surface_area * point_density)
        pcd = mesh.sample_points_poisson_disk(min_points)
        self.geometry_points = pcd.points

        # Allow "max_depth" to be minimum octree voxel size
        root_length = 1
        max_depth = round(np.log(root_length/min_voxel)/np.log(2))

        # Get root node dimension
        bbox_x = max(self.geometry_points[:, 0]) - min(self.geometry_points[:, 0])
        bbox_y = max(self.geometry_points[:, 1]) - min(self.geometry_points[:, 1])
        bbox_z = max(self.geometry_points[:, 2]) - min(self.geometry_points[:, 2])
        self.root_length = max([bbox_x, bbox_y, bbox_z])

        # Translate Part to bounding box center
        x_c = min(self.geometry_points[:, 0]) + bbox_x/2
        y_c = min(self.geometry_points[:, 1]) + bbox_y/2
        z_c = min(self.geometry_points[:, 2]) + bbox_z/2
        pcd.translate([-x_c, -y_c, -z_c])
        self.geometry_points = pcd.points

        # Verify Bounding Box Cetner
        x_c = min(self.geometry_points[:, 0]) + bbox_x/2
        y_c = min(self.geometry_points[:, 1]) + bbox_y/2
        z_c = min(self.geometry_points[:, 2]) + bbox_z/2
        print(x_c, y_c, z_c)

if __name__ == "__main__":

    file = 'GenerativeBracket.stl'
    stree = SphereTree(file, 2.5)

