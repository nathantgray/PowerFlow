import numpy as np


def mismatch(v, d, y, pq, pv, psched, qsched):
	# This function was written by Nathan Gray
	# This function calculates mismatches between the real and reactive power
	# injections in a system vs. the scheduled injections.
	# power system network.
	# Arguments:
	# d: list of voltage phase angles in system
	# V: list of voltage magnitudes in system
	# Y: Ybus matrix for system
	# PQ: list of PQ busses
	# PV: list of PV busses
	# Psched, Qsched: list of real, reactive power injections
	# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
	# S = P + jQ
	s = (v*np.exp(1j*d))*np.conj(y.dot(v*np.exp(1j*d)))
	pcalc = s[np.concatenate((pv,pq))].real
	qcalc = s[pq].imag
	dp = psched-pcalc
	dq = qsched-qcalc
	mis = np.transpose(np.concatenate((dp, dq)))
	return mis, pcalc, qcalc
