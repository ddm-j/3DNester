cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[np.float_t, ndim=3] collision_pairs(double[:,:] a, double[:,:] b):

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

cpdef np.ndarray[np.float_t, ndim=3] all_collision_pairs(double[:, :, :, :] arr):

    # Type Variables
    cdef int i, j, k, m, n, s
    m = arr.shape[0]
    n = arr.shape[2]*arr.shape[2]
    s = m*n
    print(m, n, s)
    cdef np.ndarray[np.float_t, ndim=3] out = np.empty((s, 2, 3), dtype=np.float)
    cdef double [:, :, :] out_view = out
    cdef double [:, :, :] temp1
    cdef double [:, :, :] temp2

    # Loop
    for i in range(m):
        # Loop through each "pair" of object data
        print(i*m*n, (i+1)*m*n)
        print(out_view[i*m:(i+1)*m+1, :, :].shape)
        out_view[i*m:(i+1)*m+1, :, :] = collision_pairs(arr[i, 0, :, :], arr[i, 1, :, :])[:]

    return out
