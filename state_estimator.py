import numpy as np
def h_calc(ps, v, d):
	v = np.transpose(v)[0]
	d = np.transpose(d)[0]
	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	p = np.real(s)
	q = np.real(s)
	pij, qij, pji, qji = ps.branch_flows(v, d)
	return np.r_[
		np.transpose([v]),
		np.transpose([p]),
		np.transpose([q]),
		np.transpose([pij]),
		np.transpose([qij]),
		np.transpose([pji]),
		np.transpose([qji])
	]


if __name__ == "__main__":
	from power_system import PowerSystem
	from numpy.linalg import inv
	import matplotlib.pyplot as plt
	case_name = "IEEE14BUS_handout.txt"
	# case_name = "2BUS.txt"
	ps = PowerSystem(case_name, sparse=False)
	v0, d0 = ps.flat_start()
	v, d, it = ps.pf_newtonraphson(v0, d0, prec=5, maxit=10, qlim=True)
	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	p = np.real(s)
	q = np.imag(s)
	pij, qij, pji, qji = ps.branch_flows(v, d)
	v_stdev = 0.003
	p_stdev = 0.002
	q_stdev = 0.005
	pij_stdev = p_stdev/2
	qij_stdev = q_stdev/2
	v_noise = np.random.normal(0, v_stdev, v.shape)
	p_noise = np.random.normal(0, p_stdev, p.shape)
	q_noise = np.random.normal(0, q_stdev, q.shape)
	pij_noise = np.random.normal(0, pij_stdev, pij.shape)
	qij_noise = np.random.normal(0, qij_stdev, qij.shape)
	pji_noise = np.random.normal(0, pij_stdev, pji.shape)
	qji_noise = np.random.normal(0, qij_stdev, qji.shape)
	v_meas = v + v_noise
	p_meas = p + p_noise
	q_meas = q + q_noise
	pij_meas = pij + pij_noise
	pji_meas = pji + pji_noise
	qij_meas = qij + qij_noise
	qji_meas = qji + qji_noise
	z = np.transpose([np.r_[v_meas, p_meas, q_meas, pij_meas, qij_meas, pji_meas, qji_meas]])
	# print(z)
	# print(pij)
	# print(qij)
	h = ps.se_h_matrix(v, d)
	m, n = h.shape
	nbus = len(ps.bus_data[:, 0])
	nbr = len(ps.branch_data[:, 0])
	# print(m)
	# print(n)
	r = np.zeros((m, m))
	for i in range(m):
		if i < nbus:
			r[i, i] = 1/v_stdev**2
		elif i < 2*nbus:
			r[i, i] = 1/p_stdev**2
		elif i < 3*nbus:
			r[i, i] = 1/q_stdev**2
		elif i < 3*nbus + nbr:
			r[i, i] = 1/pij_stdev**2
		elif i < 3*nbus + 2*nbr:
			r[i, i] = 1/qij_stdev**2
		elif i < 3*nbus + 3*nbr:
			r[i, i] = 1/pij_stdev**2
		elif i < 3*nbus + 4*nbr:
			r[i, i] = 1/qij_stdev**2

	d_est = np.transpose([d0])
	v_est = np.transpose([v0])
	for it in range(5):
		h = ps.se_h_matrix(v_est, d_est)
		z_mismatch =  z-h_calc(ps, v_est, d_est)
		dx =np.dot(inv(np.dot(np.dot(np.transpose(h), inv(r)), h)), np.dot(np.dot(np.transpose(h), inv(r)), z_mismatch))
		print("max dx = ", max(abs(dx)))
		if max(abs(dx)) < 0.01:
			print("iteration: ", it)
			break
		d_est = d_est + dx[0:len(d0)]
		v_est = v_est + dx[len(d0)-1:-1]
	print(d - np.transpose(d_est)[0])
	print(v - np.transpose(v_est)[0])



