import numpy as np
import time

nx, ny, ns = 512, 512, 9
nrep = 10

storage = [(nx, ny, ns), (ns, nx, ny)]

mod2test = [#"d2q9_nxnyns_vec", "d2q9_nsnxny_vec",
            "d2q9_nxnyns_vec_pythran", 
            "d2q9_nxnyns_loop_pythran",
            "d2q9_nsnxny_vec_pythran", 
            "d2q9_nsnxny_loop_pythran", 
            "d2q9_nxnyns_vec_numba", 
            "d2q9_nxnyns_loop_numba", 
            "d2q9_nsnxny_vec_numba", 
            "d2q9_nsnxny_loop_numba",
            "d2q9_nxnyns_vec_parakeet", "d2q9_nxnyns_loop_parakeet", 
            "d2q9_nsnxny_vec_parakeet", "d2q9_nsnxny_loop_parakeet",
            "d2q9_nxnyns_cython", "d2q9_nsnxny_cython"]

f2test = [#"m2f(m, f)",
          #"f2m(f, m)",
          #"transport(f)",
          #"relaxation(m)",
          #"periodic_bc(f)",
          "one_time_step(f, m)"]

for func in f2test:
    tab = np.zeros(len(mod2test))
    mlups = np.zeros(len(mod2test))
    ind = 0

    for mod in mod2test:
        exec "import " + mod
    
        if mod.find("nxnyns") != -1:
            s = 0
        else:
            s = 1
        m = np.zeros(storage[s])
        f = np.zeros(storage[s])

        t = time.time()
        for i in xrange(nrep):
            exec mod + '.' + func 
        tab[ind] = time.time() - t
        mlups[ind] = nx*ny*nrep/tab[ind]*1e-6
        ind += 1
    
    sort = np.argsort(tab)

    print func
    print '*'*20
    for s in sort:
        print "{0:30} {1:.6f} {2:.3f} {3:.3f}".format(mod2test[s], tab[s], tab[s]/tab[sort[0]], mlups[s])
    print 
