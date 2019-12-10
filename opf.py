import numpy as np
from copy import deepcopy
from power_system import PowerSystem
from numpy.linalg import inv
from crout_reorder import mat_solve


if __name__ == "__main__":
	# case_name = "IEEE14BUS.txt"
	case_name = "IEEE14BUS_handout.txt"
	ps = PowerSystem(case_name, sparse=True)
	v0, d0 = ps.flat_start()
	v, d, it = ps.pf_newtonraphson(v0, d0, prec=3, qlim=False, verbose=False)

	# 1. Solve ED without PF for P_total = 259 MW
	# 2. From base case PF, used P_total = P1 + P2
	# 3. Solve OPF with u = P2
	# 4 Solve OPF with P12 <= 5MW using penalty function
	# (use p.u. with base of 100MW)

	# 1. Solve ED without PF for P_total = 259 MW
	P_total = 259
	a1 = np.array([[8], [6.4]])  # F1(P1) = a2[0]*P1**2 + a1[0]*P1,   F2(P2) = a2[1]*P2**2 + a1[1]*P2
	a2 = np.array([[0.004], [0.0048]])
	mat = np.r_[np.c_[(np.eye(2) * 2*a2), np.array([[-1],[-1]])],np.array([[1, 1, 0]])]
	lossless_ED = mat_solve(mat, np.r_[-a1,[[P_total]]])
	print(lossless_ED)

	# 2. From base case PF, used P_total = P1 + P2
	pg2 = lossless_ED[1]
	dgdx = ps.jacobian_full(v, d)
	n = dgdx.shape[1]
	dfdx = (2 * a2[1] * pg2 + a1[1]) * (np.array([dgdx.full[1, :]]))
	print(dfdx)
	print(dgdx.full.T ** -1 @ dfdx)