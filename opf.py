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
	v, d, _ = ps.pf_newtonraphson(v0, d0, prec=4, qlim=False, verbose=False)

	# 1. Solve ED without PF for P_total = 259 MW
	# 2. From base case PF, used P_total = P1 + P2
	# 3. Solve OPF with u = P2
	# 4 Solve OPF with P12 <= 5MW using penalty function
	# (use p.u. with base of 100MW)

	# 1. Solve ED without PF for P_total = 259 MW
	if case_name == "IEEE14BUS_handout.txt" or case_name == "IEEE14BUS.txt":
		u_index = np.array([0])
		n = len(u_index)
		slack_index = 0
		P_total = 265/ps.p_base
		a1 = np.array([[8], [6.4]])*ps.p_base  # F1(P1) = a2[0]*P1**2 + a1[0]*P1,   F2(P2) = a2[1]*P2**2 + a1[1]*P2
		a2 = np.array([[0.004], [0.0048]])*ps.p_base**2
		p12_max = 50/ps.p_base
	if case_name == "3BUS_OPF.txt":
		u_index = np.array([0, 1])
		n = len(u_index)
		slack_index = 0
		P_total = 952/ps.p_base
		a1 = np.array([[1], [1], [1]])*ps.p_base  # F1(P1) = a2[0]*P1**2 + a1[0]*P1,   F2(P2) = a2[1]*P2**2 + a1[1]*P2
		a2 = np.array([[0.0625], [0.0125], [0.025]])*ps.p_base**2
		p12_max = 1000/ps.p_base

	mat = np.r_[np.c_[(np.eye(n+1) * 2*a2), -np.ones((n+1, 1))], np.array([np.r_[np.ones(n+1), 0]])]
	lossless_ED = mat_solve(mat, np.r_[-a1,[[P_total]]])
	print(lossless_ED)

	# 2. From base case PF, used P_total = P1 + P2
	ps.psched[u_index] = lossless_ED[u_index + 1] - ps.bus_data[u_index + 1, ps.busLoadMW]/ps.p_base
	v, d, _ = ps.pf_newtonraphson(v, d, prec=4, qlim=False, verbose=False)

	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	p = s.real
	q = s.imag

	# 3. Solve OPF with u = P2
	step = 1/ps.p_base
	u = np.array([lossless_ED[u_index + 1]]).T # initial guess for P2
	pg = p + ps.bus_data[:, ps.busLoadMW]/ps.p_base
	pg1 = p[0] + ps.bus_data[0, ps.busLoadMW]/ps.p_base
	for it in range(1000):
		p12 = ps.pij_flow(d, v, 0, 1)
		dgdx = -ps.pf_jacobian(v, d, ps.pq, v_mul=False)
		m = dgdx.shape[0]
		dp1dx = ps.dslack_dx(v, d, ps.pq)
		# dfdx = (2*a2[slack_index]*pg1 + a1[slack_index])*(np.array([dgdx.full[slack_index, :]]).T)
		dfdx = (2*a2[slack_index]*pg1 + a1[slack_index])*dp1dx
		if p12 > p12_max:
			dp12dx = np.zeros((m, 1))
			dp12dx[0] = -v[0]*v[1]*np.abs(ps.y_bus[0, 1])*np.sin(np.angle(ps.y_bus[0, 1]) + d[1])
			dfdx += 2*(p12 - p12_max)*dp12dx*ps.p_base**2/10
		# print(dfdx)
		l = -np.array([mat_solve(dgdx.full.T, dfdx)]).T
		# print('lambda', l)
		dgdu = np.zeros((m, n))
		dgdu[u_index, u_index] = np.array([[1]])
		# print('dgdu', dgdu)
		# dfdu = np.zeros((l.shape))
		dfdu = a1[u_index+1] + 2 * a2[u_index+1] * u
		# print('dfdu', dfdu)
		dldu = dfdu + (dgdu.T @ l)[u_index]
		u = u - step*dldu[u_index]
		s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
		p = s.real
		pg1 = p[0]
		ps.psched[u_index] = u - ps.bus_data[u_index + 1, ps.busLoadMW] / ps.p_base
		v, d, _ = ps.pf_newtonraphson(v, d, prec=4, qlim=False, verbose=False)
		print('dldu', dldu[u_index], u, p12)
		if np.max(np.abs(dldu[u_index])) < 0.05:
			p12 = ps.pij_flow(d, v, 0, 1)
			print('success!')
			print('pg1', pg1)
			print('u', u)
			print('p12', p12)
			print('l', l)
			print('Pd', sum(ps.bus_data[:, ps.busLoadMW]))
			break

	pg = np.array([[pg1], [np.ravel(u)[0]]])
	print('pg', pg)
	c = (a2*pg**2 + a1*pg)
	print('costs', c)
	print('total cost', sum(c))
