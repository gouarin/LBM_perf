#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free 

cdef void m2f_loc(double *m, double *f) nogil:
    cdef:
        double c0 = 1./6
        double c1 = 1./9  
        double c2 = 1./18
        double c3 = 1./36
        double c4 = 1./12

    f[0] = c1*m[0] - c1*m[3] + c1*m[4]
    f[1] = c1*m[0] + c0*m[1] - c3*m[3] - c2*m[4] - c0*m[5] + 0.25*m[7]
    f[2] = c1*m[0] + c0*m[2] - c3*m[3] - c2*m[4] - c0*m[6] - 0.25*m[7]
    f[3] = c1*m[0] - c0*m[1] - c3*m[3] - c2*m[4] + c0*m[5] + 0.25*m[7]
    f[4] = c1*m[0] - c0*m[2] - c3*m[3] - c2*m[4] + c0*m[6] - 0.25*m[7]
    f[5] = c1*m[0] + c0*m[1] + c0*m[2] + c2*m[3] + c3*m[4] + c4*m[5] + c4*m[6] + 0.25*m[8]
    f[6] = c1*m[0] - c0*m[1] + c0*m[2] + c2*m[3] + c3*m[4] - c4*m[5] + c4*m[6] - 0.25*m[8]
    f[7] = c1*m[0] - c0*m[1] - c0*m[2] + c2*m[3] + c3*m[4] - c4*m[5] - c4*m[6] + 0.25*m[8]
    f[8] = c1*m[0] + c0*m[1] - c0*m[2] + c2*m[3] + c3*m[4] + c4*m[5] - c4*m[6] - 0.25*m[8]

cdef void f2m_loc(double *f, double *m) nogil:
    m[0] = f[0] + f[1] + f[2] + f[3] + f[4] + f[5] + f[6] + f[7] + f[8]
    m[1] = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
    m[2] = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
    m[3] = -4.*f[0] - f[1] - f[2] - f[3] - f[4] + 2.*f[5] + 2.*f[6] + 2.*f[7] + 2.*f[8]
    m[4] = 4.*f[0] - 2.*f[1] - 2.*f[2] - 2.*f[3] - 2.*f[4] + f[5] + f[6] + f[7] + f[8]
    m[5] = -2.*f[1] + 2.*f[3] + f[5] - f[6] - f[7] + f[8]
    m[6] = -2.*f[2] + 2.*f[4] + f[5] + f[6] - f[7] - f[8]
    m[7] = f[1] - f[2] + f[3] - f[4]
    m[8] = f[5] - f[6] + f[7] - f[8]

cdef void getf(double[:, :, ::1] f, double *floc, int i, int j) nogil:
    floc[0] = f[i, j, 0]
    floc[1] = f[i-1, j, 1]
    floc[2] = f[i, j-1, 2]
    floc[3] = f[i+1, j, 3]
    floc[4] = f[i, j+1, 4]
    floc[5] = f[i-1, j-1, 5]
    floc[6] = f[i+1, j-1, 6]
    floc[7] = f[i+1, j+1, 7]
    floc[8] = f[i-1, j+1, 8]

cdef void setf(double[:, :, ::1] f, double *floc, int i, int j) nogil:
    f[j, i, 0] = floc[0]
    f[j, i, 1] = floc[1]
    f[j, i, 2] = floc[2]
    f[j, i, 3] = floc[3]
    f[j, i, 4] = floc[4]
    f[j, i, 5] = floc[5]
    f[j, i, 6] = floc[6]
    f[j, i, 7] = floc[7]
    f[j, i, 8] = floc[8]

cdef void relaxation_loc(double *m) nogil:
    m[3] += 1.1312217194570136*(-2*m[0] + 3.0*m[1]*m[1] + 3.0*m[2]*m[2] - m[3])
    m[4] += 1.1312217194570136*(m[0] + 1.5*m[1]*m[1] + 1.5*m[2]*m[2] - m[4])
    m[5] += 1.1312217194570136*(-1.0*m[1] - m[5])
    m[6] += 1.1312217194570136*(-1.0*m[2] - m[6])
    m[7] += 1.8573551263001487*(1.0*m[1]*m[1] - 1.0*m[2]*m[2] - m[7])
    m[8] += 1.8573551263001487*(1.0*m[1]*m[2] - m[8])

cdef periodic_bc(double[:, :, ::1] f):
    cdef:
        int i, j, k
        int nx = f.shape[0]
        int ny = f.shape[1]
        int ns = f.shape[2]
	
    for i in xrange(nx):
        for k in xrange(ns):
            f[i, 0, k] = f[i, ny-2, k]
            f[i, ny-1, k] = f[i, 1, k]
    for j in xrange(ny):
        for k in xrange(ns):
            f[0, j, k] = f[nx-2, j, k]
            f[nx-1, j, k] = f[1, j, k]

def one_time_step(double[:, :, ::1] f1, double[:, :, ::1] f2):
    cdef:
        double *floc
        double *mloc
        int i, j
        int nx = f1.shape[0]
        int ny = f1.shape[1]
        
    periodic_bc(f1)
    with nogil, parallel():
        floc = <double *> malloc(9*sizeof(double))
        mloc = <double *> malloc(9*sizeof(double))

        for i in prange(1, nx-1, schedule='dynamic'):
            for j in prange(1, ny-1, schedule='dynamic'):
                getf(f1, floc, i, j)
                f2m_loc(floc, mloc)
                relaxation_loc(mloc)
                m2f_loc(mloc, floc)
                setf(f2, floc, i, j)
        
        free(floc)
        free(mloc)