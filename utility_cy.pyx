cimport numpy as np
import numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dgemm
from libc.math cimport sqrt, abs
import ctypes as ct
from libcpp cimport bool


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float_t, ndim=3] collision_point_pairs(double[:,:] a, double[:,:] b):

    # Type Variables
    cdef int i, j, cnt, a_shape, b_shape
    a_shape = a.shape[0]
    b_shape = b.shape[0]
    cdef np.ndarray[np.float_t, ndim=3] arr = np.empty((a_shape*b_shape, 2, 3), dtype=np.float)
    cdef double [:, :, :] arr_view = arr
    cnt = 0

    # Loop
    for i in range(a_shape):
        for j in range(b_shape):
            # Double Slice
            arr_view[cnt,0,:] = a[i]
            arr_view[cnt,1,:] = b[j]

            cnt += 1

    return arr

#@cython.boundscheck(False)
#@cython.wraparound(False)
cpdef np.ndarray[np.float_t, ndim=2] collision_pairs(double[:] a, n):

    # Type Variables
    cdef int i, j, cnt, a_shape, b_shape
    a_shape = a.shape[0]
    cdef np.ndarray[np.float_t, ndim=2] arr = np.empty((n+1, 2), dtype=np.float)
    cdef double [:, :] arr_view = arr
    cnt = 0

    # Loop
    for i in range(a_shape):
        print(i)
        for j in range(a_shape):
            if i == j:
                continue

            # Double Slice
            arr_view[cnt,0] = i
            arr_view[cnt,1] = j

            cnt += 1

    return arr


cpdef remove_pairs(int index, int n, np.ndarray[np.float_t, ndim=2] pairs, np.ndarray[np.float_t, ndim=1] hist):

    cdef int i, pair_shape, ind
    cdef double a, b
    cdef np.ndarray[np.float_t, ndim=2] new_pairs = np.empty((n, 2), dtype=np.float)
    cdef np.ndarray[np.float_t, ndim=1] new_hist = np.empty((n,), dtype=np.float)
    cdef double[:, :] old_pair_view = pairs
    cdef double[:] old_hist_view = hist
    cdef double[:, :] pair_view = new_pairs
    cdef double[:] hist_view = new_hist
    print(n)
    ind = index
    pair_shape = pairs.shape[0]
    for i in range(pair_shape):
        print(i)
        a = old_pair_view[i, 0]
        b = old_pair_view[i, 1]

        # Skip this index if we are removing this pair
        if a == ind or b == ind:
            continue

        # Update indices
        if a > ind:
            pair_view[i, 0] -= 1
        elif b > ind:
            pair_view[i, 1] -= 1
        else:
            pair_view[i, 0] = a
            pair_view[i, 1] = b

        # Update History
        hist_view[i] = old_hist_view[i]

    return new_pairs, new_hist


@cython.boundscheck(False)
@cython.wraparound(False)
def remove_from_collision_array(int index, int n, double[:,:] arr):

    cdef int i, j

    # Shift the diagonal matrix
    for i in range(index, n-1):
        for j in range(i, n-1):
                arr[i, j] = arr[i+1, j+1]

    # Shift the rop row
    for j in range(index, n-1):
        arr[0, j] = arr[0, j+1]

    # Set Column column n-1 to zero
    for i in range(n):
        arr[i, n-1] = 0

    return np.asarray(arr)

#@cython.boundscheck(False)
#@cython.wraparound(False)
def remove_from_array(int index, int n, double[:] arr):

    cdef int i, j

    # Shift the diagonal matrix
    for i in range(index, n-1):
        arr[i] = arr[i+1]

    # Set Column column n-1 to zero
    arr[n-1] = 0

    return np.asarray(arr)

@cython.boundscheck(False)
@cython.wraparound(False)
def part_collisions(int[:,:] pairs,
                    int[:] inds,
                    double[:, :] center_points,
                    double[:, :] a,
                    double[::1, :] b,
                    double[:,:] coll_arr,
                    double sphere,
                    double leaf,
                    double interval):

    cdef:
        # Variables
        int n_pairs = len(pairs)
        int n_parts = center_points.shape[0]
        int n_inds = inds.shape[0]
        int collisions
        np.ndarray[np.float_t, ndim=2] c1 = np.empty((4, b.shape[1]), dtype=float, order='F')
        np.ndarray[np.float_t, ndim=2] c2 = np.empty((4, b.shape[1]), dtype=float, order='F')
        int o1, o2, m, n, k, lda, ldb, ldc

        # Views
        double[::1,:] c1_view = c1
        double[::1,:] c2_view = c2

        # Pointers for BLAS
        double* a1_0
        double* a2_0
        double* b0 = &b[0, 0]
        double* c1_0 = &c1_view[0, 0]
        double* c2_0 = &c2_view[0, 0]
        char* transa = 'n'
        char* transb = 'n'

    # Setting shit up for BLAS
        double beta = 0.0
        double alpha = 1.0
    lda = a.shape[0]
    ldb = b.shape[0]
    ldc = c1_view.shape[0]
    m = 4
    n = b.shape[1]
    k = 4

    # BROAD PHASE ALGORITHM
    # Begin pair-wise iteration
    for i in range(n_pairs):
        o1 = pairs[i, 0]
        o2 = pairs[i, 1]
        # Test center point collision
        if sphere_check(center_points[o1],
                        center_points[o2],
                        sphere+interval):

            # NARROW PHASE ALGORITHM
            # Get Affine Matrix Pointers
            a1_0 = &a[:, o1*4:(o1+1)*4][0, 0]
            a2_0 = &a[:, o2*4:(o2+1)*4][0, 0]

            # Perform transforms using DGEMM (BLASSSSSter)
            dgemm(transa, transb, &m, &n, &k, &alpha, a1_0, &lda, b0, &ldb, &beta, c1_0, &ldc)
            dgemm(transa, transb, &m, &n, &k, &alpha, a2_0, &lda, b0, &ldb, &beta, c2_0, &ldc)

            # We've got c1 & c2, transformed point arrays. Now calculate collision count between them.
            collisions = collision_counter(c1, c2, 2*leaf+interval)

            # Update our collision data
            if collisions != 0:
                coll_arr[o1, o2] = collisions

            '''if collisions != 0:
                print('Collisions for pair: {0},{1}'.format(o1,o2))
                #print(np.asarray(c1_view[:,0:5]))
                #print(np.asarray(c2_view[:,0:5]))
                print('Total collision vs. total possible: {0}, {1}'.format(collisions, n**2))
                print(np.asarray(inds))
                print('Column index: {}'.format(get_index(inds,o1)))'''

        else:
            # No collision between center points
            coll_arr[o1, o2] = 0

    # Get the sum
    total_collisions = sum_collisions(n_parts, coll_arr)

    return coll_arr, total_collisions



@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint sphere_check(double[:] u, double[:] v, double dist):

    cdef double delta = 0
    cdef int x_shape = u.shape[0]
    cdef bint collision

    for i in range(x_shape-1):
        delta += (u[i] - v[i])*(u[i] - v[i])
    delta = sqrt(delta)
    collision = delta < dist

    return collision

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int collision_counter(double[:,:] u, double[:,:] v, double dist):

    cdef int cnt = 0
    cdef int x_shape = u.shape[1]

    for i in range(x_shape):
        for j in range(x_shape):
            if sphere_check(u[:,i],v[:,j], dist):
                cnt += 1
    return cnt

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int get_index(int[:] arr, int val):

    cdef int x_shape = arr.shape[0]
    for i in range(x_shape):
        if arr[i] == val:
            return i

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double sum_collisions(int n, double[:,:] arr):

    cdef double total = 0

    # Shift the diagonal matrix
    for i in range(n):
        for j in range(i, n):
            if arr[i,j] < 0.000000001:
                total += 0
            else:
                total += arr[i, j]

    return total