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
        int nx = m.shape[0]
        int ny = m.shape[1]
        int ns = m.shape[2]

    for i in xrange(nx):
        for j in xrange(ny):
            f[i, j, 0] = c1*m[i, j, 0] - c1*m[i, j, 3] + c1*m[i, j, 4]
            f[i, j, 1] = c1*m[i, j, 0] + c0*m[i, j, 1] - c3*m[i, j, 3] - c2*m[i, j, 4] - c0*m[i, j, 5] + 0.25*m[i, j, 7]
            f[i, j, 2] = c1*m[i, j, 0] + c0*m[i, j, 2] - c3*m[i, j, 3] - c2*m[i, j, 4] - c0*m[i, j, 6] - 0.25*m[i, j, 7]
            f[i, j, 3] = c1*m[i, j, 0] - c0*m[i, j, 1] - c3*m[i, j, 3] - c2*m[i, j, 4] + c0*m[i, j, 5] + 0.25*m[i, j, 7]
            f[i, j, 4] = c1*m[i, j, 0] - c0*m[i, j, 2] - c3*m[i, j, 3] - c2*m[i, j, 4] + c0*m[i, j, 6] - 0.25*m[i, j, 7]
            f[i, j, 5] = c1*m[i, j, 0] + c0*m[i, j, 1] + c0*m[i, j, 2] + c2*m[i, j, 3] + c3*m[i, j, 4] + c4*m[i, j, 5] + c4*m[i, j, 6] + 0.25*m[i, j, 8]
            f[i, j, 6] = c1*m[i, j, 0] - c0*m[i, j, 1] + c0*m[i, j, 2] + c2*m[i, j, 3] + c3*m[i, j, 4] - c4*m[i, j, 5] + c4*m[i, j, 6] - 0.25*m[i, j, 8]
            f[i, j, 7] = c1*m[i, j, 0] - c0*m[i, j, 1] - c0*m[i, j, 2] + c2*m[i, j, 3] + c3*m[i, j, 4] - c4*m[i, j, 5] - c4*m[i, j, 6] + 0.25*m[i, j, 8]
            f[i, j, 8] = c1*m[i, j, 0] + c0*m[i, j, 1] - c0*m[i, j, 2] + c2*m[i, j, 3] + c3*m[i, j, 4] + c4*m[i, j, 5] - c4*m[i, j, 6] - 0.25*m[i, j, 8]

cpdef f2m(double[:, :, ::1] f, double[:, :, ::1] m):
    cdef:
        int i, j, k
        int nx = m.shape[0]
        int ny = m.shape[1]
        int ns = m.shape[2]
        double floc[9]

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(ns):
                floc[k] = f[i, j, k]

            m[i, j, 0] = floc[0] + floc[1] + floc[2] + floc[3] + floc[4] + floc[5] + floc[6] + floc[7] + floc[8]
            m[i, j, 1] = floc[1] - floc[3] + floc[5] - floc[6] - floc[7] + floc[8]
            m[i, j, 2] = floc[2] - floc[4] + floc[5] + floc[6] - floc[7] - floc[8]
            m[i, j, 3] = -4.*floc[0] - floc[1] - floc[2] - floc[3] - floc[4] + 2.*floc[5] + 2.*floc[6] + 2.*floc[7] + 2.*floc[8]
            m[i, j, 4] = 4.*floc[0] - 2.*floc[1] - 2.*floc[2] - 2.*floc[3] - 2.*floc[4] + floc[5] + floc[6] + floc[7] + floc[8]
            m[i, j, 5] = -2.*floc[1] + 2.*floc[3] + floc[5] - floc[6] - floc[7] + floc[8]
            m[i, j, 6] = -2.*floc[2] + 2.*floc[4] + floc[5] + floc[6] - floc[7] - floc[8]
            m[i, j, 7] = floc[1] - floc[2] + floc[3] - floc[4]
            m[i, j, 8] = floc[5] - floc[6] + floc[7] - floc[8]

def transport(double[:, :, ::1] f):
    cdef: 
        int i, j
        int nx = f.shape[0]
        int ny = f.shape[1]

    for i in xrange(nx-1, 0, -1):
        for j in xrange(ny):
            f[i, j, 1] = f[i-1, j, 1]

    for i in xrange(nx):
        for j in xrange(ny-1, 0, -1):
            f[i, j, 2] = f[i, j-1, 2]

    for i in xrange(nx-1):
        for j in xrange(ny):
            f[i, j, 3] = f[i+1, j, 3]

    for i in xrange(nx):
        for j in xrange(ny-1):
            f[i, j, 4] = f[i, j+1, 4]

    for i in xrange(nx-1, 0, -1):
        for j in xrange(ny-1, 0, -1):
            f[i, j, 5] = f[i-1, j-1, 5]

    for i in xrange(nx-1):
        for j in xrange(ny-1, 0, -1):
            f[i, j, 6] = f[i+1, j-1, 6]

    for i in xrange(nx-1):
        for j in xrange(ny-1):
            f[i, j, 7] = f[i+1, j+1, 7]
            
    for i in xrange(nx-1, 0, -1):
        for j in xrange(ny-1):
            f[i, j, 8] = f[i-1, j+1, 8]

def equilibrium(double[:, :, ::1] m):
    cdef:
        int i, j, k
        int nx = m.shape[0]
        int ny = m.shape[1]
        int ns = m.shape[2]
        double mloc[9]
        
    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(3):
                mloc[k] = m[i, j, k]

            m[i, j, 3] = -2*mloc[0] + 3.0*mloc[1]*mloc[1] + 3.0*mloc[2]*mloc[2]
            m[i, j, 4] = mloc[0] + 1.5*mloc[1]*mloc[1] + 1.5*mloc[2]*mloc[2]
            m[i, j, 5] = -mloc[1]
            m[i, j, 6] = -mloc[2]
            m[i, j, 7] = mloc[1]*mloc[1] - 1.0*mloc[2]*mloc[2]
            m[i, j, 8] = mloc[1]*mloc[2]
    
cpdef relaxation(double[:, :, ::1] m):
    cdef:
        int i, j, k
        int nx = m.shape[0]
        int ny = m.shape[1]
        int ns = m.shape[2]
        double mloc[9]

    for i in xrange(nx):
        for j in xrange(ny):
            for k in xrange(ns):
                mloc[k] = m[i, j, k]
            m[i, j, 3] += 1.1312217194570136*(-2*mloc[0] + 3.0*mloc[1]*mloc[1] + 3.0*mloc[2]*mloc[2] - mloc[3])
            m[i, j, 4] += 1.1312217194570136*(mloc[0] + 1.5*mloc[1]*mloc[1] + 1.5*mloc[2]*mloc[2] - mloc[4])
            m[i, j, 5] += 1.1312217194570136*(-1.0*mloc[1] - mloc[5])
            m[i, j, 6] += 1.1312217194570136*(-1.0*mloc[2] - mloc[6])
            m[i, j, 7] += 1.8573551263001487*(1.0*mloc[1]*mloc[1] - 1.0*mloc[2]*mloc[2] - mloc[7])
            m[i, j, 8] += 1.8573551263001487*(1.0*mloc[1]*mloc[2] - mloc[8])

cpdef periodic_bc(double[:, :, ::1] f):
    cdef:
        int i, j, k
        int nx = f.shape[0]
        int ny = f.shape[1]
        int ns = f.shape[2]

    for k in xrange(ns):
        for i in xrange(nx):
            f[i, 0, k] = f[i, ny-2, k]
            f[i, ny-1, k] = f[i, 1, k]
        for j in xrange(ny):
            f[0, j, k] = f[nx-2, j, k]
            f[nx-1, j, k] = f[1, j, k]

def one_time_step(double[:, :, ::1] f, double[:, :, ::1] m):
    periodic_bc(f)
    transport(f)
    f2m(f, m)
    relaxation(m)
    m2f(m, f)

