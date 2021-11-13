import open3d as o3d
import numpy as np

# Set Parameters
res = 2.5
maxVoxCA = (res**2)*np.sqrt(2)
pointDensity = 15/maxVoxCA

# Load Mesh
mesh = o3d.io.read_triangle_mesh('meshes/GenerativeBracket.stl')
surfaceArea = mesh.get_surface_area()
minPoints = round(surfaceArea*pointDensity)

# Create Point Cloud
pcd = mesh.sample_points_poisson_disk(minPoints)
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(minPoints, 3)))

#o3d.visualization.draw_geometries([pcd])

bbox = pcd.get_axis_aligned_bounding_box().get_extent()
l = max(bbox)

levels = round(np.log(l/res)/np.log(2))


octree = o3d.geometry.Octree(max_depth=2)
octree.convert_from_point_cloud(pcd, size_expand=0.01)


def rotation_matrix(x, y, z, degrees=True):

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

def callback(node, node_info):

    print(node_info)

def traverse(node, level=0):
    print("Entering tree level: {}".format(level))

    if hasattr(node,"children"):

        for child in node.children:
            print(type(child))
            print(child)
            traverse(child, level=level+1)





# Root note: octree.root_node
# Children: octree.root_node.children, returns list of children nodes with notes

o3d.visualization.draw_geometries([octree])
#traverse(octree.root_node)

rotation_vector = [90, 0, 0]

rot_mat = octree.get_rotation_matrix_from_xyz(list(map(np.deg2rad, rotation_vector)))

octree.rotate(rot_mat)

#traverse(octree.root_node)

o3d.visualization.draw_geometries([octree])