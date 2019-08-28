import numpy as np
from mismatch import mismatch
from read_testcase import readCase
from makeYbus import makeYbus
from PF_Jacobian import PF_Jacobian
from numpy.linalg import inv


def PF_NewtonRaphson(v, d, y, pq, pv, psched, qsched, prec=2, maxit=5):

	# Uses Newton-Raphson method to solve the power-flow of a power system.
	# Also capable of Q limiting.
	# Written by Nathan Gray
	# Arguments:
	# d: list of voltage phase angles in system
	# V: list of voltage magnitudes in system
	# Y: Ybus matrix for system
	# PQ: list of PQ busses
	# PV: list of PV busses
	# Psched, Qsched: list of real, reactive power injections
	# Qd: Reactive power demand (Qd and Qlim are not used in this version)
	# Qlim: array of Q limits- 1st column is maximums 2nd is minimums
	# prec: program finishes when all mismatches < 10^-abs(prec)
	maxit = 4

	n = np.shape(y)[0]
	#Ng = length(PV);

	gMaxQ = []
	gMinQ = []
	## Newton Raphson
	for it in range(maxit):
		# Calculate Jacobian
		j, j11, j21, j12, j22 = PF_Jacobian(v, d, y, pq)
		# Calculate Mismatches
		mis, pcalc, qcalc = mismatch(v, d, y, pq, pv, psched, qsched)
		# Calculate update values
		dx =np.dot(inv(j), mis)
		# Update angles: d_(n+1) = d_n + dd
		d[np.sort(np.concatenate((pv, pq)))] = d[np.sort(np.concatenate((pv, pq)))] + dx[:n - 1]
		# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
		v[pq] = v[pq]*(1+dx[n-1:n+pq.size-1])
		# Check error
		mis, pcalc, qcalc = mismatch(v, d, y, pq, pv, psched, qsched)
		if max(mis) < 10**-abs(prec):
			print("Newton Raphson completed in ", it, " iterations.")
			break

		if it >= maxit:
			print("Max iterations reached, ", it, ".")

	return v, d, it

