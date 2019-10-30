
from power_system import PowerSystem
from crout_reorder import *
import time

case_name = "IEEE14BUS_handout.txt"
ps = PowerSystem(case_name, sparse=True)
y = ps.y_bus
v, d = ps.flat_start()
b, p_calc, q_calc = ps.mismatch(v, d, y, ps.pq, ps.pvpq, ps.psched, ps.qsched)
a = ps.pf_jacobian(v, d, ps.pq)
x = mat_solve(a, b)

# Test 1: no ordering
print("\nStart test-1 with no ordering---------------------------")
time_start = time.perf_counter()
q = sparse_crout(a)
print("crout time = ", time.perf_counter() - time_start)
print(q.full(dtype=bool).astype(int))
time_start = time.perf_counter()
x = lu_solve(q, b)
print("solve time = ", time.perf_counter() - time_start)
print("x=\n", x)
print("b check =\n", a.dot(x) - b)
print("alpha=", q.alpha())
print("beta=", q.beta())
print("alpha + beta = ", q.alpha() + q.beta())

# Test 2: Tinny-0 ordering
print("\nStart test-2 with Tinny-0 ordering---------------------------")
order0 = tinny0(a)
print("order 0: ", order0)
time_start = time.perf_counter()
q = sparse_crout(a, order=tinny0(a))
print("crout time = ", time.perf_counter() - time_start)
print(q.full(dtype=bool).astype(int))
time_start = time.perf_counter()
x = lu_solve(q, b, order=order0)
print("solve time = ", time.perf_counter() - time_start)
print("x=\n", x)
print("b check =\n", a.dot(x) - b)
print("alpha=", q.alpha())
print("beta=", q.beta())
print("alpha + beta = ", q.alpha() + q.beta())
print("degrees: ", node_degrees(a))
ndegs = node_degrees(a)
print("order: ", order0)

# Test 3: Tinny-1 ordering
print("\nStart test-3 with Tinny-1 ordering---------------------------")
order1 = tinny1(a)
print("order 1: ", order1)
time_start = time.perf_counter()
q = sparse_crout(a, order=tinny1(a))
print("crout time = ", time.perf_counter() - time_start)
print(q.full(dtype=bool).astype(int))
time_start = time.perf_counter()
x = lu_solve(q, b, order=order1)
print("solve time = ", time.perf_counter() - time_start)
print("x=\n", x)
print("b check =\n", a.dot(x) - b)
print("alpha=", q.alpha())
print("beta=", q.beta())
print("alpha + beta = ", q.alpha() + q.beta())
print("degrees: ", node_degrees(a))
ndegs = node_degrees(a)
print("order: ", order1)
