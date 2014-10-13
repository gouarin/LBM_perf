#pythran export m2f(float64 [][][], float64 [][][]) 
def m2f(m, f):
    c0 = 1./6
    c1 = 1./9
    c2 = 1./18
    c3 = 1./36
    c4 = 1./12

    ns, nx, ny = m.shape

    for i in xrange(nx):
        for j in xrange(ny):
            f[0, i, j] = c1*m[0, i, j] - c1*m[3, i, j] + c1*m[4, i, j]
            f[1, i, j] = c1*m[0, i, j] + c0*m[1, i, j] - c3*m[3, i, j] - c2*m[4, i, j] - c0*m[5, i, j] + 0.25*m[7, i, j]
            f[2, i, j] = c1*m[0, i, j] + c0*m[2, i, j] - c3*m[3, i, j] - c2*m[4, i, j] - c0*m[6, i, j] - 0.25*m[7, i, j]
            f[3, i, j] = c1*m[0, i, j] - c0*m[1, i, j] - c3*m[3, i, j] - c2*m[4, i, j] + c0*m[5, i, j] + 0.25*m[7, i, j]
            f[4, i, j] = c1*m[0, i, j] - c0*m[2, i, j] - c3*m[3, i, j] - c2*m[4, i, j] + c0*m[6, i, j] - 0.25*m[7, i, j]
            f[5, i, j] = c1*m[0, i, j] + c0*m[1, i, j] + c0*m[2, i, j] + c2*m[3, i, j] + c3*m[4, i, j] + c4*m[5, i, j] + c4*m[6, i, j] + 0.25*m[8, i, j]
            f[6, i, j] = c1*m[0, i, j] - c0*m[1, i, j] + c0*m[2, i, j] + c2*m[3, i, j] + c3*m[4, i, j] - c4*m[5, i, j] + c4*m[6, i, j] - 0.25*m[8, i, j]
            f[7, i, j] = c1*m[0, i, j] - c0*m[1, i, j] - c0*m[2, i, j] + c2*m[3, i, j] + c3*m[4, i, j] - c4*m[5, i, j] - c4*m[6, i, j] + 0.25*m[8, i, j]
            f[8, i, j] = c1*m[0, i, j] + c0*m[1, i, j] - c0*m[2, i, j] + c2*m[3, i, j] + c3*m[4, i, j] + c4*m[5, i, j] - c4*m[6, i, j] - 0.25*m[8, i, j]

#pythran export f2m(float64 [][][], float64 [][][]) 
def f2m(f, m):
    ns, nx, ny = m.shape

    for i in xrange(nx):
        for j in xrange(ny):
            m[0, i, j] = f[0, i, j] + f[1, i, j] + f[2, i, j] + f[3, i, j] + f[4, i, j] + f[5, i, j] + f[6, i, j] + f[7, i, j] + f[8, i, j]
            m[1, i, j] = f[1, i, j] - f[3, i, j] + f[5, i, j] - f[6, i, j] - f[7, i, j] + f[8, i, j]
            m[2, i, j] = f[2, i, j] - f[4, i, j] + f[5, i, j] + f[6, i, j] - f[7, i, j] - f[8, i, j]
            m[3, i, j] = -4.*f[0, i, j] - f[1, i, j] - f[2, i, j] - f[3, i, j] - f[4, i, j] + 2.*f[5, i, j] + 2.*f[6, i, j] + 2.*f[7, i, j] + 2.*f[8, i, j]
            m[4, i, j] = 4.*f[0, i, j] - 2.*f[1, i, j] - 2.*f[2, i, j] - 2.*f[3, i, j] - 2.*f[4, i, j] + f[5, i, j] + f[6, i, j] + f[7, i, j] + f[8, i, j]
            m[5, i, j] = -2.*f[1, i, j] + 2.*f[3, i, j] + f[5, i, j] - f[6, i, j] - f[7, i, j] + f[8, i, j]
            m[6, i, j] = -2.*f[2, i, j] + 2.*f[4, i, j] + f[5, i, j] + f[6, i, j] - f[7, i, j] - f[8, i, j]
            m[7, i, j] = f[1, i, j] - f[2, i, j] + f[3, i, j] - f[4, i, j]
            m[8, i, j] = f[5, i, j] - f[6, i, j] + f[7, i, j] - f[8, i, j]

#pythran export transport(float64 [][][]) 
def transport(f):
    ns, nx, ny = f.shape

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

#pythran export equilibrium(float64 [][][]) 
def equilibrium(m):
    ns, nx, ny = m.shape

    for i in xrange(nx):
        for j in xrange(ny):
            m[3, i, j] = -2*m[0, i, j] + 3.0*m[1, i, j]**2 + 3.0*m[2, i, j]**2
            m[4, i, j] = m[0, i, j] + 1.5*m[1, i, j]**2 + 1.5*m[2, i, j]**2
            m[5, i, j] = -1.0*m[1, i, j]
            m[6, i, j] = -1.0*m[2, i, j]
            m[7, i, j] = 1.0*m[1, i, j]**2 - 1.0*m[2, i, j]**2
            m[8, i, j] = 1.0*m[1, i, j]*m[2, i, j]
    
#pythran export relaxation(float64 [][][]) 
def relaxation(m):
    ns, nx, ny = m.shape

    for i in xrange(nx):
        for j in xrange(ny):
            m[3, i, j] += 1.1312217194570136*(-2*m[0, i, j] + 3.0*m[1, i, j]**2 + 3.0*m[2, i, j]**2 - m[3, i, j])
            m[4, i, j] += 1.1312217194570136*(m[0, i, j] + 1.5*m[1, i, j]**2 + 1.5*m[2, i, j]**2 - m[4, i, j])
            m[5, i, j] += 1.1312217194570136*(-1.0*m[1, i, j] - m[5, i, j])
            m[6, i, j] += 1.1312217194570136*(-1.0*m[2, i, j] - m[6, i, j])
            m[7, i, j] += 1.8573551263001487*(1.0*m[1, i, j]**2 - 1.0*m[2, i, j]**2 - m[7, i, j])
            m[8, i, j] += 1.8573551263001487*(1.0*m[1, i, j]*m[2, i, j] - m[8, i, j])

#pythran export periodic_bc(float64 [][][]) 
def periodic_bc(f):
    ns, nx, ny = f.shape

    for k in xrange(ns):
        for j in xrange(ny):
            f[k, 0, j] = f[k, nx-2, j]
            f[k, nx-1, j] = f[k, 1, j]
    for k in xrange(ns):
        for i in xrange(nx):
            f[k, i, 0] = f[k, i, ny-2]
            f[k, i, ny-1] = f[k, i, 1]

#pythran export one_time_step(float64 [][][], float64 [][][]) 
def one_time_step(f, m):
    periodic_bc(f)
    transport(f)
    f2m(f, m)
    relaxation(m)
    m2f(m, f)
