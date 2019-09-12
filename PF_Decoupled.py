import numpy as np
from mismatch import mismatch
from crout import mat_solve
from copy import deepcopy

def pf_decoupled(v, d, y, pq, pvpq, psched, qsched, prec=2, maxit=4):
	v = deepcopy(v)
	d = deepcopy(d)
	n = np.shape(y)[0]
	# Decoupled Power Flow
	it = 0
	# bd = np.zeros(y.shape)
	# bd[np.where(y)] = 1/(np.imag(1/y[np.where(y)]))
	# bd = bd[pvpq, :][:, pvpq]
	bd = -y.imag[pvpq, :][:, pvpq]
	bv = -y.imag[pq, :][:, pq]
	for it in range(maxit):
		# Calculate Mismatches
		mis = mismatch(v, d, y, pq, pvpq, psched, qsched)[0]
		# Check error
		if max(abs(mis)) < 10**-abs(prec):
			print("Decoupled Power Flow completed in ", it, " iterations.")
			return v, d, it
		d[pvpq] = d[pvpq] + mat_solve(bd, mis[0:len(pvpq)]/v[pvpq])
		v[pq] = v[pq] + mat_solve(bv, mis[len(pvpq):]/v[pq])

	print("Max iterations reached, ", it, ".")