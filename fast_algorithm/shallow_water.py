import numpy as np
import matplotlib.pyplot as plt
import d2q9_nxnyns_cython as d2q9

def init_sol(f, m, x, y):
    m[:, :, 0] = rhoo * np.ones((y.size, x.size)) + deltarho * ((x-0.5*(xmin+xmax))**2 + (y-0.5*(ymin+ymax))**2 < 0.5**2)
    m[:, :, 1] = np.zeros((y.size, x.size))
    m[:, :, 2] = np.zeros((y.size, x.size))
    equilibrium(m)
    m2f(m, f)

def m2f(m, f):
    f[:, :, 0] =  0.1111111111111111*m[:, :, 0] - 0.1111111111111111*m[:, :, 3] + 0.1111111111111111*m[:, :, 4]
    f[:, :, 1] =  0.1111111111111111*m[:, :, 0] + 0.1666666666666667*m[:, :, 1] - 0.0277777777777778*m[:, :, 3] - 0.0555555555555556*m[:, :, 4] - 0.1666666666666667*m[:, :, 5] + 0.2500000000000000*m[:, :, 7]
    f[:, :, 2] =  0.1111111111111111*m[:, :, 0] + 0.1666666666666667*m[:, :, 2] - 0.0277777777777778*m[:, :, 3] - 0.0555555555555556*m[:, :, 4] - 0.1666666666666667*m[:, :, 6] - 0.2500000000000000*m[:, :, 7]
    f[:, :, 3] =  0.1111111111111111*m[:, :, 0] - 0.1666666666666667*m[:, :, 1] - 0.0277777777777778*m[:, :, 3] - 0.0555555555555556*m[:, :, 4] + 0.1666666666666667*m[:, :, 5] + 0.2500000000000000*m[:, :, 7]
    f[:, :, 4] =  0.1111111111111111*m[:, :, 0] - 0.1666666666666667*m[:, :, 2] - 0.0277777777777778*m[:, :, 3] - 0.0555555555555556*m[:, :, 4] + 0.1666666666666667*m[:, :, 6] - 0.2500000000000000*m[:, :, 7]
    f[:, :, 5] =  0.1111111111111111*m[:, :, 0] + 0.1666666666666667*m[:, :, 1] + 0.1666666666666667*m[:, :, 2] + 0.0555555555555556*m[:, :, 3] + 0.0277777777777778*m[:, :, 4] + 0.0833333333333333*m[:, :, 5] + 0.0833333333333333*m[:, :, 6] + 0.2500000000000000*m[:, :, 8]
    f[:, :, 6] =  0.1111111111111111*m[:, :, 0] - 0.1666666666666667*m[:, :, 1] + 0.1666666666666667*m[:, :, 2] + 0.0555555555555556*m[:, :, 3] + 0.0277777777777778*m[:, :, 4] - 0.0833333333333333*m[:, :, 5] + 0.0833333333333333*m[:, :, 6] - 0.2500000000000000*m[:, :, 8]
    f[:, :, 7] =  0.1111111111111111*m[:, :, 0] - 0.1666666666666667*m[:, :, 1] - 0.1666666666666667*m[:, :, 2] + 0.0555555555555556*m[:, :, 3] + 0.0277777777777778*m[:, :, 4] - 0.0833333333333333*m[:, :, 5] - 0.0833333333333333*m[:, :, 6] + 0.2500000000000000*m[:, :, 8]
    f[:, :, 8] =  0.1111111111111111*m[:, :, 0] + 0.1666666666666667*m[:, :, 1] - 0.1666666666666667*m[:, :, 2] + 0.0555555555555556*m[:, :, 3] + 0.0277777777777778*m[:, :, 4] + 0.0833333333333333*m[:, :, 5] - 0.0833333333333333*m[:, :, 6] - 0.2500000000000000*m[:, :, 8]

def f2m(f, m):
    m[:, :, 0] =  f[:, :, 0] + f[:, :, 1] + f[:, :, 2] + f[:, :, 3] + f[:, :, 4] + f[:, :, 5] + f[:, :, 6] + f[:, :, 7] + f[:, :, 8]
    m[:, :, 1] =  f[:, :, 1] - f[:, :, 3] + f[:, :, 5] - f[:, :, 6] - f[:, :, 7] + f[:, :, 8]
    m[:, :, 2] =  f[:, :, 2] - f[:, :, 4] + f[:, :, 5] + f[:, :, 6] - f[:, :, 7] - f[:, :, 8]
    m[:, :, 3] =  - 4.0000000000000000*f[:, :, 0] - f[:, :, 1] - f[:, :, 2] - f[:, :, 3] - f[:, :, 4] + 2.0000000000000000*f[:, :, 5] + 2.0000000000000000*f[:, :, 6] + 2.0000000000000000*f[:, :, 7] + 2.0000000000000000*f[:, :, 8]
    m[:, :, 4] =  4.0000000000000000*f[:, :, 0] - 2.0000000000000000*f[:, :, 1] - 2.0000000000000000*f[:, :, 2] - 2.0000000000000000*f[:, :, 3] - 2.0000000000000000*f[:, :, 4] + f[:, :, 5] + f[:, :, 6] + f[:, :, 7] + f[:, :, 8]
    m[:, :, 5] =  - 2.0000000000000000*f[:, :, 1] + 2.0000000000000000*f[:, :, 3] + f[:, :, 5] - f[:, :, 6] - f[:, :, 7] + f[:, :, 8]
    m[:, :, 6] =  - 2.0000000000000000*f[:, :, 2] + 2.0000000000000000*f[:, :, 4] + f[:, :, 5] + f[:, :, 6] - f[:, :, 7] - f[:, :, 8]
    m[:, :, 7] =  f[:, :, 1] - f[:, :, 2] + f[:, :, 3] - f[:, :, 4]
    m[:, :, 8] =  f[:, :, 5] - f[:, :, 6] + f[:, :, 7] - f[:, :, 8]

def equilibrium(m):
    m[:, :, 3] = -2*m[:, :, 0] + 3.0*m[:, :, 1]**2 + 3.0*m[:, :, 2]**2
    m[:, :, 4] = m[:, :, 0] + 1.5*m[:, :, 1]**2 + 1.5*m[:, :, 2]**2
    m[:, :, 5] = -1.0*m[:, :, 1]
    m[:, :, 6] = -1.0*m[:, :, 2]
    m[:, :, 7] = 1.0*m[:, :, 1]**2 - 1.0*m[:, :, 2]**2
    m[:, :, 8] = 1.0*m[:, :, 1]*m[:, :, 2]

if __name__ == '__main__':
    Taille = 2.
    xmin, xmax, ymin, ymax = -0.5*Taille, 0.5*Taille, -0.5*Taille, 0.5*Taille
    dx = 1./256 # spatial step
    x = np.linspace(xmin - .5*dx, xmax + .5*dx, int((xmax - xmin + .5*dx)/dx))
    x = x[np.newaxis, :]
    y = np.linspace(ymin - .5*dx, ymax + .5*dx, int((ymax - ymin + .5*dx)/dx))
    y = y[:, np.newaxis]

    rhoo = 1.
    deltarho = 1.

    nx, ny, ns = x.size, y.size, 9
    nrep = 100

    m = np.zeros((nx, ny, ns), dtype=np.float64)
    f1 = np.zeros((nx, ny, ns), dtype=np.float64)
    f2 = np.zeros((nx, ny, ns), dtype=np.float64)

    init_sol(f1, m, x, y)

    #from pyevtk.hl import imageToVTK 
    #save = np.zeros((nx, ny, 1))

    import time
    t = time.time()

    for i in xrange(nrep):
        d2q9.one_time_step(f1, f2)
        tmp = f1
        f1 = f2
        f2 = tmp

        # f2m(f1, m)
        # save[:, :, 0] = m[:, :, 0]
        # imageToVTK("./data/image_{0}".format(i), pointData = {"rho" : save} )
    t = time.time() - t
    print t, nx*ny*nrep/t/1e6


    # #import mlab
    # #mlab.surf(m[:, :, 0], warp_scale='auto')

    f2m(f1, m)
    plt.imshow(m[:, :, 0])
    plt.show()