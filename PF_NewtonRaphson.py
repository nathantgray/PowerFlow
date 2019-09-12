import numpy as np
from mismatch import mismatch
from PF_Jacobian import pf_jacobian
from numpy.linalg import inv
from crout import mat_solve
from copy import deepcopy


def pf_newtonraphson(v, d, y, pq, pv, psched, qsched, qlim, prec=2, maxit=4):
	# Uses Newton-Raphson method to solve the power-flow of a power system.
	# Written by Nathan Gray
	# Arguments:
	# v: list of voltage magnitudes in system
	# d: list of voltage phase angles in system
	# y: Ybus matrix for system
	# pq: list of PQ buses
	# pv: list of PV buses
	# psched, qsched: list of real, reactive power injections
	# qlim: [qmax, qmin]
	# prec: program finishes when all mismatches < 10^-abs(prec)
	pvpq = np.sort(np.concatenate((pv, pq)))  # list of indices of non-slack buses
	v = deepcopy(v)
	d = deepcopy(d)

	n = np.shape(y)[0]
	# Newton Raphson
	it = 0
	for it in range(maxit):
		# Calculate Mismatches
		mis = mismatch(v, d, y, pq, pvpq, psched, qsched)[0]
		# Check error
		if max(abs(mis)) < 10**-abs(prec):
			print("Newton Raphson completed in ", it, " iterations.")
			return v, d, it
		check_limits(v, d, y, pq, pv, qlim)
		# Calculate Jacobian
		j = pf_jacobian(v, d, y, pq)
		# Calculate update values
		dx = mat_solve(j, mis)
		# Update angles: d_(n+1) = d_n + dd
		d[pvpq] = d[pvpq] + dx[:n - 1]
		# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
		v[pq] = v[pq]*(1+dx[n-1:n+pq.size-1])

	print("Max iterations reached, ", it, ".")

def check_limits(v, d, y, pq, pv, qlim):
	# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
	s = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
	q_calc = s.imag
	gbus = deepcopy(pv)
	q_min = [min(lim) for lim in qlim]
	q_max = [max(lim) for lim in qlim]
	maxlim_gbus_indexes = np.where(np.array([max(lim) <= q_calc[i] for i, lim in enumerate(qlim)])[gbus])[0]
	minlim_gbus_indexes = np.where(np.array([min(lim) <= q_calc[i] for i, lim in enumerate(qlim)])[gbus])[0]
	if maxlim_gbus_indexes in pv:
		newpv = np.delete(pv, maxlim_gbus_indexes)
		newpq = np.sort(np.concatenate(pq, maxlim_gbus_indexes))

	print(newpv)

