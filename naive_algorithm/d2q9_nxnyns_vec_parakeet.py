import d2q9_nxnyns_vec as d2q9
from parakeet import jit

m2f = jit(d2q9.m2f)
f2m = jit(d2q9.f2m)
transport = jit(d2q9.transport)
relaxation = jit(d2q9.relaxation)
periodic_bc = jit(d2q9.periodic_bc)

@jit
def one_time_step(f, m):
    periodic_bc(f)
    transport(f)
    f2m(f, m)
    relaxation(m)
    m2f(m, f)
