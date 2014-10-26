import d2q9_nxnyns as d2q9
from numba import cuda, float64, float32
import numpy as np
import matplotlib.pyplot as plt

m2f_loc = cuda.jit('void (f8[::1], f8[::1])', device=True)(d2q9.m2f_loc)
f2m_loc = cuda.jit('void (f8[::1], f8[::1])', device=True)(d2q9.f2m_loc)
getf = cuda.jit('void (f8[:,:,:], f8[::1], i8, i8)', device=True)(d2q9.getf)
setf = cuda.jit('void (f8[:,:,:], f8[::1], i4, i4)', device=True)(d2q9.setf)
relaxation_loc = cuda.jit('void (f8[::1])', device=True)(d2q9.relaxation_loc)

@cuda.jit('void(f8[:,:,:])')
def periodic_bc(f1):
    nx, ny, ns = f1.shape
    i, j = cuda.grid(2)
    
    if i == 0:
        for k in xrange(ns):
            f1[i, j, k] = f1[nx - 2, j, k]
    if i == nx-1:
        for k in xrange(ns):
            f1[i, j, k] = f1[1, j, k]

    if j == 0:
        for k in xrange(ns):
            f1[i, j, k] = f1[i, ny - 2, k]
    if j == ny-1:
        for k in xrange(ns):
            f1[i, j, k] = f1[i, 1, k]

@cuda.jit('void(f8[:,:,:], f8[:,:,:])')
def one_time_step(f1, f2):
    nx, ny, ns = f1.shape
    floc = cuda.local.array(9, dtype=float64)  
    mloc = cuda.local.array(9, dtype=float64)    
    i, j = cuda.grid(2)

    if i>0 and i<nx-1 and j>0 and j<ny-1:
        getf(f1, floc, i, j)
        f2m_loc(floc, mloc)
        relaxation_loc(mloc)
        m2f_loc(mloc, floc)
        setf(f2, floc, i, j)
