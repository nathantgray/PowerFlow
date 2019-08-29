from numpy import exp, conj, concatenate


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
	# pvpq: list of PV and pq buses
	# psched, qsched: list of real, reactive power injections

	# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
	s = (v*exp(1j*d))*conj(y.dot(v*exp(1j*d)))
	# S = P + jQ
	pcalc = s[pvpq].real
	qcalc = s[pq].imag
	dp = psched-pcalc
	dq = qsched-qcalc
	mis = concatenate((dp, dq))
	return mis, pcalc, qcalc
