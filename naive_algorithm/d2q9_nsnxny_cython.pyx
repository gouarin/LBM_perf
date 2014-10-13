#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

cpdef m2f(double[:, :, ::1] m, double[:, :, ::1] f):
    cdef:
        double c0 = 1./6
        double c1 = 1./9  
        double c2 = 1./18
        double c3 = 1./36
        double c4 = 1./12
        int i, j, k
        int nx = m.shape[1]
        int ny = m.shape[2]
        int ns = m.shape[0]
        double mloc[9]

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(ns):
                mloc[k] = m[k, i, j]

            f[0, i, j] = c1*mloc[0] - c1*mloc[3] + c1*mloc[4]
            f[1, i, j] = c1*mloc[0] + c0*mloc[1] - c3*mloc[3] - c2*mloc[4] - c0*mloc[5] + 0.25*mloc[7]
            f[2, i, j] = c1*mloc[0] + c0*mloc[2] - c3*mloc[3] - c2*mloc[4] - c0*mloc[6] - 0.25*mloc[7]
            f[3, i, j] = c1*mloc[0] - c0*mloc[1] - c3*mloc[3] - c2*mloc[4] + c0*mloc[5] + 0.25*mloc[7]
            f[4, i, j] = c1*mloc[0] - c0*mloc[2] - c3*mloc[3] - c2*mloc[4] + c0*mloc[6] - 0.25*mloc[7]
            f[5, i, j] = c1*mloc[0] + c0*mloc[1] + c0*mloc[2] + c2*mloc[3] + c3*mloc[4] + c4*mloc[5] + c4*mloc[6] + 0.25*mloc[8]
            f[6, i, j] = c1*mloc[0] - c0*mloc[1] + c0*mloc[2] + c2*mloc[3] + c3*mloc[4] - c4*mloc[5] + c4*mloc[6] - 0.25*mloc[8]
            f[7, i, j] = c1*mloc[0] - c0*mloc[1] - c0*mloc[2] + c2*mloc[3] + c3*mloc[4] - c4*mloc[5] - c4*mloc[6] + 0.25*mloc[8]
            f[8, i, j] = c1*mloc[0] + c0*mloc[1] - c0*mloc[2] + c2*mloc[3] + c3*mloc[4] + c4*mloc[5] - c4*mloc[6] - 0.25*mloc[8]

cpdef f2m(double[:, :, ::1] f, double[:, :, ::1] m):
    cdef:
        int i, j, k
        int nx = m.shape[1]
        int ny = m.shape[2]
        int ns = m.shape[0]
        double floc[9]

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(ns):
                floc[k] = f[k, i, j]

            m[0, i, j] = floc[0] + floc[1] + floc[2] + floc[3] + floc[4] + floc[5] + floc[6] + floc[7] + floc[8]
            m[1, i, j] = floc[1] - floc[3] + floc[5] - floc[6] - floc[7] + floc[8]
            m[2, i, j] = floc[2] - floc[4] + floc[5] + floc[6] - floc[7] - floc[8]
            m[3, i, j] = -4.*floc[0] - floc[1] - floc[2] - floc[3] - floc[4] + 2.*floc[5] + 2.*floc[6] + 2.*floc[7] + 2.*floc[8]
            m[4, i, j] = 4.*floc[0] - 2.*floc[1] - 2.*floc[2] - 2.*floc[3] - 2.*floc[4] + floc[5] + floc[6] + floc[7] + floc[8]
            m[5, i, j] = -2.*floc[1] + 2.*floc[3] + floc[5] - floc[6] - floc[7] + floc[8]
            m[6, i, j] = -2.*floc[2] + 2.*floc[4] + floc[5] + floc[6] - floc[7] - floc[8]
            m[7, i, j] = floc[1] - floc[2] + floc[3] - floc[4]
            m[8, i, j] = floc[5] - floc[6] + floc[7] - floc[8]

def transport(double[:, :, ::1] f):
    cdef: 
        int i, j
        int nx = f.shape[1]
        int ny = f.shape[2]

    for i in xrange(nx-1, 0, -1):
        for j in xrange(ny):
            f[1, i, j] = f[1, i-1, j]
            
    for i in xrange(nx):
        for j in xrange(ny-1, 0, -1):
            f[2, i, j] = f[2, i, j-1]

    for i in xrange(nx-1):
        for j in xrange(ny):
            f[3, i, j] = f[3, i+1, j]

    for i in xrange(nx):
        for j in xrange(ny-1):
            f[4, i, j] = f[4, i, j+1]

    for i in xrange(nx-1, 0, -1):
        for j in xrange(ny-1, 0, -1):
            f[5, i, j] = f[5, i-1, j-1]

    for i in xrange(nx-1):
        for j in xrange(ny-1, 0, -1):
            f[6, i, j] = f[6, i+1, j-1]

    for i in xrange(nx-1):
        for j in xrange(ny-1):
            f[7, i, j] = f[7, i+1, j+1]
            
    for i in xrange(nx-1, 0, -1):
        for j in xrange(ny-1):
            f[8, i, j] = f[8, i-1, j+1]


def equilibrium(double[:, :, ::1] m):
    cdef:
        int i, j, k
        int nx = m.shape[1]
        int ny = m.shape[2]
        int ns = m.shape[0]
        double mloc[9]

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(3):
                mloc[k] = m[k, i, j]

            m[3, i, j] = -2*mloc[0] + 3.0*mloc[1]*mloc[1] + 3.0*mloc[2]*mloc[2]
            m[4, i, j] = mloc[0] + 1.5*mloc[1]*mloc[1] + 1.5*mloc[2]*mloc[2]
            m[5, i, j] = -mloc[1]
            m[6, i, j] = -mloc[2]
            m[7, i, j] = mloc[1]*mloc[1] - 1.0*mloc[2]*mloc[2]
            m[8, i, j] = mloc[1]*mloc[2]
    
def relaxation(double[:, :, ::1] m):
    cdef:
        int i, j, k
        int nx = m.shape[1]
        int ny = m.shape[2]
        int ns = m.shape[0]
        double mloc[9]

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(ns):
                mloc[k] = m[k, i, j]
            m[3, i, j] += 1.1312217194570136*(-2*mloc[0] + 3.0*mloc[1]*mloc[1] + 3.0*mloc[2]*mloc[2] - mloc[3])
            m[4, i, j] += 1.1312217194570136*(mloc[0] + 1.5*mloc[1]*mloc[1] + 1.5*mloc[2]*mloc[2] - mloc[4])
            m[5, i, j] += 1.1312217194570136*(-1.0*mloc[1] - mloc[5])
            m[6, i, j] += 1.1312217194570136*(-1.0*mloc[2] - mloc[6])
            m[7, i, j] += 1.8573551263001487*(1.0*mloc[1]*mloc[1] - 1.0*mloc[2]*mloc[2] - mloc[7])
            m[8, i, j] += 1.8573551263001487*(1.0*mloc[1]*mloc[2] - mloc[8])

cpdef periodic_bc(double[:, :, ::1] f):
    cdef:
        int i, j, k
        int nx = f.shape[1]
        int ny = f.shape[2]
        int ns = f.shape[0]

    for k in xrange(ns):
        for i in xrange(nx):
            f[k, i, 0] = f[k, i, ny-2]
            f[k, i, ny-1] = f[k, i, 1]
        for j in xrange(ny):
            f[k, 0, j] = f[k, nx-2, j]
            f[k, nx-1, j] = f[k, 1, j]


def one_time_step(double[:, :, ::1] f, double[:, :, ::1] m):
    periodic_bc(f)
    transport(f)
    f2m(f, m)
    relaxation(m)
    m2f(m, f)

