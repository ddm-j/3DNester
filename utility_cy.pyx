cimport numpy as np
import numpy as np
cimport cython

MAX_PARTS = 5000

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

@cython.boundscheck(False)
@cython.wraparound(False)
def remove_from_array(int index, int n, double[:] arr):

    cdef int i, j

    # Shift the diagonal matrix
    for i in range(index, n-1):
        arr[i] = arr[i+1]

    # Set Column column n-1 to zero
    arr[n-1] = 0

    return np.asarray(arr)