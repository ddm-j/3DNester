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