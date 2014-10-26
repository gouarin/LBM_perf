import d2q9_nxnyns as d2q9
from numba import autojit, jit
import numpy as np

m2f_loc = jit('void (f8[:], f8[:])', nopython=True)(d2q9.m2f_loc)
f2m_loc = jit('void (f8[:], f8[:])', nopython=True)(d2q9.f2m_loc)
getf = jit('void (f8[:,:,:], f8[:], i4, i4)', nopython=True)(d2q9.getf)
setf = jit('void (f8[:,:,:], f8[:], i4, i4)', nopython=True)(d2q9.setf)
relaxation_loc = jit('void (f8[:])', nopython=True)(d2q9.relaxation_loc)
periodic_bc  = jit('void (f8[:,:,:])', nopython=True)(d2q9.periodic_bc)

@jit('void (f8[:,:,:], f8[:,:,:])')
def one_time_step(f1, f2):
    nx, ny, ns = f1.shape
    floc = np.zeros(ns)    
    mloc = np.zeros(ns)    
    
    periodic_bc(f1)
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            getf(f1, floc, i, j)
            f2m_loc(floc, mloc)
            relaxation_loc(mloc)
            m2f_loc(mloc, floc)
            setf(f2, floc, i, j)
  
if __name__ == '__main__':      
     m2f_loc.inspect_types()
