import numpy as np


def mismatch(v, d, y, pq, pvpq, psched, qsched):
	# This function was written by Nathan Gray
	# This function calculates mismatches between the real and reactive power
	# injections in a system vs. the scheduled injections.
	# power system network.
	# Arguments:
	# v: list of voltage magnitudes in system
	# d: list of voltage phase angles in system
	# y: Ybus matrix for system
	# pq: list of PQ buses
	# pv: list of PV buses
	# psched, qsched: list of real, reactive power injections

	# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
	s = (v*np.exp(1j*d))*np.conj(y.dot(v*np.exp(1j*d)))
	# S = P + jQ
	pcalc = s[pvpq].real
	qcalc = s[pq].imag
	dp = psched-pcalc
	dq = qsched-qcalc
	mis = np.concatenate((dp, dq))
	return mis, pcalc, qcalc
