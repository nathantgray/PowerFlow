import numpy as np
from copy import deepcopy
from power_system import PowerSystem
from numpy.linalg import inv
from crout_reorder import mat_solve


if __name__ == "__main__":
	# case_name = "IEEE14BUS.txt"
	case_name = "IEEE14BUS_handout.txt"
	# case_name = "3BUS_OPF.txt"
	ps = PowerSystem(case_name, sparse=True)
	v0, d0 = ps.flat_start()
	v, d, _ = ps.pf_newtonraphson(v0, d0, prec=3, qlim=False, verbose=False)

	# 1. Solve ED without PF for P_total = 259 MW
	# 2. From base case PF, used P_total = P1 + P2
	# 3. Solve OPF with u = P2
	# 4 Solve OPF with P12 <= 5MW using penalty function
	# (use p.u. with base of 100MW)

	# 1. Solve ED without PF for P_total = 259 MW
	u_index = np.array([0])
	n = len(u_index)
	slack_index = 0
	P_total = 259/ps.p_base
	a1 = np.array([[8], [6.4]])*ps.p_base  # F1(P1) = a2[0]*P1**2 + a1[0]*P1,   F2(P2) = a2[1]*P2**2 + a1[1]*P2
	a2 = np.array([[0.004], [0.0048]])*ps.p_base**2

	# u_index = np.array([0, 1])
	# n = len(u_index)
	# slack_index = 0
	# P_total = 952/ps.p_base
	# a1 = np.array([[1], [1], [1]])*ps.p_base  # F1(P1) = a2[0]*P1**2 + a1[0]*P1,   F2(P2) = a2[1]*P2**2 + a1[1]*P2
	# a2 = np.array([[0.0625], [0.0125], [0.025]])*ps.p_base**2

	mat = np.r_[np.c_[(np.eye(n+1) * 2*a2), -np.ones((n+1, 1))], np.array([np.r_[np.ones(n+1), 0]])]
	lossless_ED = mat_solve(mat, np.r_[-a1,[[P_total]]])
	print(lossless_ED)

	# 2. From base case PF, used P_total = P1 + P2
	ps.psched[u_index] = lossless_ED[u_index + 1] - ps.bus_data[u_index + 1, ps.busLoadMW]/ps.p_base
	v, d, _ = ps.pf_newtonraphson(v, d, prec=3, qlim=False, verbose=False)

	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	p = s.real
	q = s.imag

	# 3. Solve OPF with u = P2
	step = 1/ps.p_base
	u = np.array([lossless_ED[u_index + 1]]).T # initial guess for P2
	pg1 = p[0]
	for it in range(100):
		# dgdx = ps.jacobian_full(v, d)
		dgdx = -ps.pf_jacobian(v, d, ps.pq, v_mul=False)
		m = dgdx.shape[0]
		dp1dx = ps.dslack_dx(v, d, ps.pq)
		# dfdx = (2*a2[slack_index]*pg1 + a1[slack_index])*(np.array([dgdx.full[slack_index, :]]).T)
		dfdx = (2*a2[slack_index]*pg1 + a1[slack_index])*dp1dx
		# print(dfdx)
		l  = -np.array([mat_solve(dgdx.full.T, dfdx)]).T
		# print('lambda', l)
		dgdu = np.zeros((m, n))
		dgdu[u_index, u_index] = np.array([[1]])
		# print('dgdu', dgdu)
		# dfdu = np.zeros((l.shape))
		dfdu = a1[u_index+1] + 2 * a2[u_index+1] * u
		# print('dfdu', dfdu)
		dldu = dfdu + (dgdu.T @ l)[u_index]
		print('dldu', dldu[u_index], u)
		u = u - step*dldu[u_index]
		s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
		p = s.real
		pg1 = p[0]
		ps.psched[u_index] = u - ps.bus_data[u_index + 1, ps.busLoadMW] / ps.p_base
		if np.max(np.abs(dldu[u_index])) < 0.001:
			print('success!')
			print('u', u)
			break

