#pythran export m2f(float64 [][][], float64 [][][]) 
def m2f(m, f):
    c0 = 1./6
    c1 = 1./9  
    c2 = 1./18
    c3 = 1./36
    c4 = 1./12
    nx, ny, ns = m.shape

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


#pythran export f2m(float64 [][][], float64 [][][]) 
def f2m(f, m):
    nx, ny, ns = m.shape

    for i in xrange(nx):
        for j in xrange(ny):
            m[i, j, 0] = f[i, j, 0] + f[i, j, 1] + f[i, j, 2] + f[i, j, 3] + f[i, j, 4] + f[i, j, 5] + f[i, j, 6] + f[i, j, 7] + f[i, j, 8]
            m[i, j, 1] = f[i, j, 1] - f[i, j, 3] + f[i, j, 5] - f[i, j, 6] - f[i, j, 7] + f[i, j, 8]
            m[i, j, 2] = f[i, j, 2] - f[i, j, 4] + f[i, j, 5] + f[i, j, 6] - f[i, j, 7] - f[i, j, 8]
            m[i, j, 3] = -4.*f[i, j, 0] - f[i, j, 1] - f[i, j, 2] - f[i, j, 3] - f[i, j, 4] + 2.*f[i, j, 5] + 2.*f[i, j, 6] + 2.*f[i, j, 7] + 2.*f[i, j, 8]
            m[i, j, 4] = 4.*f[i, j, 0] - 2.*f[i, j, 1] - 2.*f[i, j, 2] - 2.*f[i, j, 3] - 2.*f[i, j, 4] + f[i, j, 5] + f[i, j, 6] + f[i, j, 7] + f[i, j, 8]
            m[i, j, 5] = -2.*f[i, j, 1] + 2.*f[i, j, 3] + f[i, j, 5] - f[i, j, 6] - f[i, j, 7] + f[i, j, 8]
            m[i, j, 6] = -2.*f[i, j, 2] + 2.*f[i, j, 4] + f[i, j, 5] + f[i, j, 6] - f[i, j, 7] - f[i, j, 8]
            m[i, j, 7] = f[i, j, 1] - f[i, j, 2] + f[i, j, 3] - f[i, j, 4]
            m[i, j, 8] = f[i, j, 5] - f[i, j, 6] + f[i, j, 7] - f[i, j, 8]

#pythran export transport(float64 [][][]) 
def transport(f):
    nx, ny, ns = f.shape

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

#pythran export equilibrium(float64 [][][]) 
def equilibrium(m):
    nx, ny, ns = m.shape

    for i in xrange(nx):
        for j in xrange(ny):
            m[i, j, 3] = -2*m[i, j, 0] + 3.0*m[i, j, 1]**2 + 3.0*m[i, j, 2]**2
            m[i, j, 4] = m[i, j, 0] + 1.5*m[i, j, 1]**2 + 1.5*m[i, j, 2]**2
            m[i, j, 5] = -1.0*m[i, j, 1]
            m[i, j, 6] = -1.0*m[i, j, 2]
            m[i, j, 7] = 1.0*m[i, j, 1]**2 - 1.0*m[i, j, 2]**2
            m[i, j, 8] = 1.0*m[i, j, 1]*m[i, j, 2]
    
#pythran export relaxation(float64 [][][]) 
def relaxation(m):
    nx, ny, ns = m.shape

    for i in xrange(nx):
        for j in xrange(ny):
            m[i, j, 3] += 1.1312217194570136*(-2*m[i, j, 0] + 3.0*m[i, j, 1]**2 + 3.0*m[i, j, 2]**2 - m[i, j, 3])
            m[i, j, 4] += 1.1312217194570136*(m[i, j, 0] + 1.5*m[i, j, 1]**2 + 1.5*m[i, j, 2]**2 - m[i, j, 4])
            m[i, j, 5] += 1.1312217194570136*(-m[i, j, 1] - m[i, j, 5])
            m[i, j, 6] += 1.1312217194570136*(-m[i, j, 2] - m[i, j, 6])
            m[i, j, 7] += 1.8573551263001487*(m[i, j, 1]**2 - m[i, j, 2]**2 - m[i, j, 7])
            m[i, j, 8] += 1.8573551263001487*(m[i, j, 1]*m[i, j, 2] - m[i, j, 8])

#pythran export periodic_bc(float64 [][][]) 
def periodic_bc(f):
    nx, ny, ns = f.shape

    for j in xrange(ny):
        for k in xrange(ns):
            f[0, j, k] = f[nx-2, j, k]
            f[nx-1, j, k] = f[1, j, k]
    for i in xrange(nx):
        for k in xrange(ns):
            f[i, 0, k] = f[i, ny-2, k]
            f[i, ny-1, k] = f[i, 1, k]

#pythran export one_time_step(float64 [][][], float64 [][][]) 
def one_time_step(f, m):
    periodic_bc(f)
    transport(f)
    f2m(f, m)
    relaxation(m)
    m2f(m, f)
