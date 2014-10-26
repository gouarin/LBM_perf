import numpy as np
import time

#nx, ny, ns = 512, 512, 9
nx, ny, ns = 1024, 1024, 9
nrep = 10

storage = [(nx, ny, ns), (ns, nx, ny)]

mod2test = ["d2q9_nsnxny_vec",
            #"d2q9_nsnxny_vec_parakeet", 
            #"d2q9_nsnxny_vec_numba", 
            #"d2q9_nsnxny_vec_pythran",
            "d2q9_nxnyns_loop_parakeet", 
            "d2q9_nxnyns_loop_numba", 
            "d2q9_nxnyns_loop_pythran",
            "d2q9_nxnyns_cython", 
            ]

# mod2test = [#"d2q9_nxnyns_vec", 
#             "d2q9_nsnxny_vec",
#             #"d2q9_nxnyns_vec_pythran", 
#             "d2q9_nxnyns_loop_pythran",
#             #"d2q9_nsnxny_vec_pythran", 
#             #"d2q9_nsnxny_loop_pythran", 
#             #"d2q9_nxnyns_vec_numba", 
#             "d2q9_nxnyns_loop_numba", 
#             #"d2q9_nsnxny_vec_numba", 
#             #"d2q9_nsnxny_loop_numba",
#             #"d2q9_nxnyns_vec_parakeet", 
#             "d2q9_nxnyns_loop_parakeet", 
#             #"d2q9_nsnxny_vec_parakeet", 
#             #"d2q9_nsnxny_loop_parakeet",
#             "d2q9_nxnyns_cython", 
#             #"d2q9_nsnxny_cython",
#             ]

f2test = ["m2f(m, f)",
          "f2m(f, m)",
          "transport(f)",
          "relaxation(m)",
          "periodic_bc(f)",
          "one_time_step(f, m)"
         ]

tab = np.zeros((len(mod2test), len(f2test)))
mlups = np.zeros((len(mod2test), len(f2test)))

for indf in xrange(len(f2test)):
    func = f2test[indf]

    for indm in xrange(len(mod2test)):
        mod = mod2test[indm]
        exec "import " + mod
    
        if mod.find("nxnyns") != -1:
            s = 0
        else:
            s = 1
            
        m = np.zeros(storage[s])
        f = np.zeros(storage[s])

        exec mod + '.' + func

        t = time.time()
        for i in xrange(nrep):
            exec mod + '.' + func

        tab[indm, indf] = (time.time() - t)/nrep
        print func, tab[indm, indf]
        mlups[indm, indf] = nx*ny/tab[indm, indf]*1e-6
          
pourcent = tab/np.sum(tab, axis=0)

for indf in xrange(len(f2test)):
    for indm in xrange(len(mod2test)):
        print "{0}; {1}; {2}; {3}; {4}".format(f2test[indf], mod2test[indm], tab[indm, indf], pourcent[indm, indf], mlups[indm, indf])
