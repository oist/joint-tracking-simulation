import numpy as np
cimport numpy as np

cdef class BrainCTRNN:
    cdef public int N
    cdef double step_size, sigma
    cdef public np.ndarray Y, I, Tau, W, Theta, G, dy_dt
    cdef object tau_range, g_range, theta_range, w_range

cdef center_cross(np.ndarray weights)
cdef sigmoid(np.ndarray x)
