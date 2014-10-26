import numpy as np
from numba import cuda
import time

nx, ny, ns = 1024, 1024, 9
nrep = 100

storage = [(nx, ny, ns), (ns, nx, ny)]

mod2test = ["d2q9_nxnyns_parakeet",
            "d2q9_nxnyns_numba", 
            "d2q9_nxnyns_pythran",
            "d2q9_nxnyns_cython",
            "d2q9_nxnyns_cython"]
            #"d2q9_nxnyns_numba_cuda"]

f2test = ["one_time_step"]

is_cuda = [False]*5 + [True]
threads = [0, 0, 0, 1, 8, 0]

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
            
        if is_cuda[indm]:
            cu_threads = 1, 128
            cu_blocks = (nx/cu_threads[0]+(0!=nx%cu_threads[0]),
                         ny/cu_threads[1]+(0!=ny%cu_threads[1]) )

            m_cpu = np.zeros(storage[s])
            f_cpu = np.zeros(storage[s])

        else:
            m = np.zeros(storage[s])
            f = np.zeros(storage[s])

        if threads[indm] == 0:
            ext = '(f, m)'
        else:
            ext = '(f, m, {0})'.format(threads[indm])

        if is_cuda[indm]:
            m = cuda.to_device(m_cpu)
            f = cuda.to_device(f_cpu)

        # t = time.time()
        # exec mod + '.' + func + ext 
        # print 'compilation time for ' + mod, time.time()-t
        t = time.time()
        if is_cuda[indm]:
            for i in xrange(nrep):
                exec mod + '.periodic_bc[cu_blocks, cu_threads](f)'
                exec mod + '.one_time_step[cu_blocks, cu_threads](f, m)'
                tmp = f
                f = m
                m = tmp
        else:
            for i in xrange(nrep):
                exec mod + '.' + func + ext 

        tab[indm, indf] = (time.time() - t)/nrep
        mlups[indm, indf] = nx*ny/tab[indm, indf]*1e-6
          
pourcent = tab/np.sum(tab, axis=0)

for indf in xrange(len(f2test)):
    for indm in xrange(len(mod2test)):
        print "{0}; {1}; {2}; {3}; {4}".format(f2test[indf], mod2test[indm], tab[indm, indf], pourcent[indm, indf], mlups[indm, indf])
