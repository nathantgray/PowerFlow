import numpy as np
from mismatch import mismatch
from PF_Jacobian import pf_jacobian
from numpy.linalg import inv


def pf_newtonraphson(v, d, y, pq, pv, psched, qsched, prec=2, maxit=4):

	# Uses Newton-Raphson method to solve the power-flow of a power system.
	# Also capable of Q limiting.
	# Written by Nathan Gray
	# Arguments:
	# d: list of voltage phase angles in system
	# V: list of voltage magnitudes in system
	# Y: Ybus matrix for system
	# PQ: list of PQ buses
	# PV: list of PV buses
	# Psched, Qsched: list of real, reactive power injections
	# prec: program finishes when all mismatches < 10^-abs(prec)
	pvpq = np.sort(np.concatenate((pv, pq)))

	n = np.shape(y)[0]
	# Newton Raphson
	it = 0
	for it in range(maxit):
		# Calculate Jacobian
		j, j11, j21, j12, j22 = pf_jacobian(v, d, y, pq)
		# Calculate Mismatches
		mis, pcalc, qcalc = mismatch(v, d, y, pq, pv, psched, qsched)
		# Calculate update values
		dx = np.dot(inv(j), mis)
		# Update angles: d_(n+1) = d_n + dd
		d[pvpq] = d[pvpq] + dx[:n - 1]
		# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
		v[pq] = v[pq]*(1+dx[n-1:n+pq.size-1])
		# Check error
		mis, pcalc, qcalc = mismatch(v, d, y, pq, pv, psched, qsched)
		if max(abs(mis)) < 10**-abs(prec):
			print("Newton Raphson completed in ", it, " iterations.")
			break

		if it >= maxit:
			print("Max iterations reached, ", it, ".")

	return v, d, it
