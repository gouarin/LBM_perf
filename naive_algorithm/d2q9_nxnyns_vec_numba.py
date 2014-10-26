import d2q9_nxnyns_vec as d2q9
from numba import double, jit, autojit

m2f = jit('void(f8[:, :, :], f8[:, :, :])')(d2q9.m2f)
f2m = jit('void(f8[:, :, :], f8[:, :, :])')(d2q9.f2m)
transport = jit('void(f8[:, :, :])')(d2q9.transport)
relaxation = jit('void(f8[:, :, :])')(d2q9.relaxation)
periodic_bc = jit('void(f8[:, :, :])')(d2q9.periodic_bc)
#one_time_step = jit('void(f8[:, :, :], f8[:, :, :])')(d2q9.one_time_step)

# m2f = autojit(d2q9.m2f)
# f2m = autojit(d2q9.f2m)
# transport = autojit(d2q9.transport)
# relaxation = autojit(d2q9.relaxation)
# periodic_bc = autojit(d2q9.periodic_bc)

@jit('void(f8[:, :, :], f8[:, :, :])')
def one_time_step(f, m):
    periodic_bc(f)
    transport(f)
    f2m(f, m)
    relaxation(m)
    m2f(m, f)
