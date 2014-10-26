import d2q9_nxnyns as d2q9
import numpy as np
from parakeet import jit
import parakeet

#parakeet.config.backend = "c"

m2f_loc = jit(d2q9.m2f_loc)
f2m_loc = jit(d2q9.f2m_loc)
getf = jit(d2q9.getf)
setf = jit(d2q9.setf)
relaxation_loc = jit(d2q9.relaxation_loc)
periodic_bc  = jit(d2q9.periodic_bc)

@jit
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

