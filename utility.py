import numpy as np
from random import gauss
import numba as nb


def split(xyzmax, xyzmin):

    # Split a node into 8 children
    # Input xyz min/max points defining a node, outputs 16 points defining child nodes
    # Credit to: https://codereview.stackexchange.com/questions/126521/python-octree-implementation

    xyzmed = (xyzmax + xyzmin) / 2

    nodes = np.array([
        [[xyzmin[0], xyzmin[1], xyzmin[2]], [xyzmed[0], xyzmed[1], xyzmed[2]]],
        [[xyzmin[0], xyzmed[1], xyzmin[2]], [xyzmed[0], xyzmax[1], xyzmed[2]]],
        [[xyzmed[0], xyzmed[1], xyzmin[2]], [xyzmax[0], xyzmax[1], xyzmed[2]]],
        [[xyzmed[0], xyzmin[1], xyzmin[2]], [xyzmax[0], xyzmed[1], xyzmed[2]]],
        [[xyzmin[0], xyzmin[1], xyzmed[2]], [xyzmed[0], xyzmed[1], xyzmax[2]]],
        [[xyzmin[0], xyzmed[1], xyzmed[2]], [xyzmed[0], xyzmax[1], xyzmax[2]]],
        [[xyzmed[0], xyzmed[1], xyzmed[2]], [xyzmax[0], xyzmax[1], xyzmax[2]]],
        [[xyzmed[0], xyzmin[1], xyzmed[2]], [xyzmax[0], xyzmed[1], xyzmax[2]]],
    ])

    return nodes


def generate_tree(points, nodes, max_depth, depth=0):
    # Recursive function. Loops down to maximum depth searching for leaf nodes. Return leafs as a list (n, 2, 3)
    # where n = number of nodes, (2, 3) correspond to xyz min/max points of each node

    leafs = []
    depth += 1

    for node in nodes:
        # Get points inside the node
        inside = check_points(points, node)

        # Do we have a point inside the node?
        if len(inside) != 0 and depth <= max_depth:

            # Split the current node, generate tree
            children = split(node[1], node[0])
            leafs += generate_tree(inside, children, max_depth, depth)

        elif len(inside) != 0:

            # We are at maximum tree depth, add node to the leaf list
            leafs.append(node)

    return leafs


def generate_spheres(nodes):
    # Reduces (n, 2, 3) node array to radius & center points (n, 3)

    # Verify that all spheres have same size
    dias = np.round(np.linalg.norm(nodes[:, 1, :] - nodes[:, 0, :], axis=1),3)
    if not np.all(dias == dias[0]):
        n = len(dias[np.where(dias != dias[0])[0]])
        raise Exception("Not all leaf sizes are equal within 3 decimal places! Number of sizes off: {}".format(n))
    else:
        r = dias[0]/2

    # Calculate sphere centers corresponding to each node
    centers = nodes[:, 0, :] + 0.5*(nodes[:, 1, :] - nodes[:, 0, :])

    return r, centers


def check_points(points, node):

    # Checks how many points inside a node. Returns points inside the node.
    # Credit to: https://stackoverflow.com/questions/21037241/how-to-determine-a-point-is-inside-or-outside-a-cube

    # Separate Node into "cube3d format"
    xmin, ymin, zmin = node[0]
    xmax, ymax, zmax = node[1]

    cube3d = np.array([[xmin, ymin, zmin],
                       [xmax, ymin, zmin],
                       [xmax, ymax, zmin],
                       [xmin, ymax, zmin],
                       [xmin, ymin, zmax],
                       [xmax, ymin, zmax],
                       [xmax, ymax, zmax],
                       [xmin, ymax, zmax]])

    b1, b2, b3, b4, t1, t2, t3, t4 = cube3d

    dir1 = (t1-b1)
    size1 = np.linalg.norm(dir1)
    dir1 = dir1 / size1

    dir2 = (b2-b1)
    size2 = np.linalg.norm(dir2)
    dir2 = dir2 / size2

    dir3 = (b4-b1)
    size3 = np.linalg.norm(dir3)
    dir3 = dir3 / size3

    cube3d_center = (b1 + t3)/2.0

    dir_vec = points - cube3d_center

    res1 = np.where((np.absolute(np.dot(dir_vec, dir1)) * 2) > size1)[0]
    res2 = np.where((np.absolute(np.dot(dir_vec, dir2)) * 2) > size2)[0]
    res3 = np.where((np.absolute(np.dot(dir_vec, dir3)) * 2) > size3)[0]

    outside = list(set().union(res1, res2, res3))

    return np.delete(points, outside, 0)


def translate(points, vector):
    # Performs relative point transformation with translation vector

    return points + vector


def rotate(points, matrix):

    return np.transpose(np.matmul(matrix, np.transpose(points)))


def rotation_matrix(vector, degrees=True):
    # Vector Utility Function
    # Takes rotations on x, y, z axes and outputs a (3, 3) rotation matrix

    # Convert degrees to radians if needed
    x, y, z = vector
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


def sphere_collision_check(vectors, d):

    # Calculate boolean collision array
    collisions = np.linalg.norm(vectors[:, 0, :]-vectors[:, 1, :], axis=1) <= d

    return collisions


def rect_collision_check(vectors, envelope, d):

    # Envelope xyz min/max
    xyzmin = envelope[0]
    xyzmax = envelope[1]

    # Calculate X, Y, & Z vectors in envelope bounds
    #collisions = np.invert(np.all(np.logical_and(vectors-d-xyzmin >= 0, vectors+d-xyzmax <= 0), axis=1))
    collisions = np.invert(np.logical_and(np.all(xyzmax-vectors-d >= 0, axis=1), np.all(xyzmin-vectors+d <= 0, axis=1)))

    return collisions


def make_rand_vector(dims, d):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    norm = [x/mag for x in vec]
    return np.array([x*d for x in norm]).reshape((dims,))

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out
