import numpy as np
from mismatch import mismatch
from PF_Jacobian import pf_jacobian
from numpy.linalg import inv


def pf_newtonraphson(v, d, y, pq, pvpq, psched, qsched, prec=2, maxit=4):
	# Uses Newton-Raphson method to solve the power-flow of a power system.
	# Written by Nathan Gray
	# Arguments:
	# v: list of voltage magnitudes in system
	# d: list of voltage phase angles in system
	# y: Ybus matrix for system
	# pq: list of PQ buses
	# pv: list of PV buses
	# psched, qsched: list of real, reactive power injections
	# prec: program finishes when all mismatches < 10^-abs(prec)

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

		# Calculate Jacobian
		j = pf_jacobian(v, d, y, pq)
		# Calculate update values
		dx = inv(j).dot(mis)
		# Update angles: d_(n+1) = d_n + dd
		d[pvpq] = d[pvpq] + dx[:n - 1]
		# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
		v[pq] = v[pq]*(1+dx[n-1:n+pq.size-1])

	print("Max iterations reached, ", it, ".")

