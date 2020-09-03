from classes.power_system_cuda import PowerSystem_CUDA
import numpy as np
from numpy import r_
import time

case_name = 'CaseFiles/Kundur_modified.txt'
ps = PowerSystem_CUDA(case_name)

def func(x):
	v = ps.bus_data[:, ps.busDesiredVolts]
	d = r_[0, x[:len(ps.pvpq)]]
	v[ps.pq] = x[len(ps.pvpq):]
	mis, _, _ = ps.mismatch(v, d, ps.y_bus, ps.pq, ps.pvpq, ps.psched, ps.qsched)
	return -mis

v, d = ps.pf()
x0 = r_[r_[d[ps.pvpq], v[ps.pq]]]
start = time.time()
j_n = ps.diff(func, x_eq=x0)
stop = time.time()
print('numerical:', stop - start)
start = time.time()
n = 100000
for i in range(n):
	j_a = ps.pf_jacobian_cuda(v, d, ps.pq)
stop = time.time()
print('algebraic average:', (stop - start)/n)

print('algebraic total for', n, 'times:', (stop - start))