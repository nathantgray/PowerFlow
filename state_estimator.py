import numpy as np
def h_calc(ps, v, d):
	v = np.transpose(v)[0]
	d = np.transpose(d)[0]
	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	p = np.real(s)
	q = np.imag(s)
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
	case_name = "IEEE14BUS_handout.txt"
	# case_name = "2BUS.txt"
	ps = PowerSystem(case_name, sparse=False)
	v0, d0 = ps.flat_start()
	# ~~~~~ Create Measurements ~~~~~
	v, d, it = ps.pf_newtonraphson(v0, d0, prec=5, maxit=10, qlim=True)
	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	p = np.real(s)
	q = np.imag(s)
	pij, qij, pji, qji = ps.branch_flows(v, d)
	v_stdev = 0.01
	p_stdev = 0.015
	q_stdev = 0.02
	pij_stdev = 0.015  # p_stdev/2
	qij_stdev = 0.02  # q_stdev/2
	np.random.seed(59)
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
	# Test with 2 bus system from Grainger book example 15.6
	if case_name == "2BUS.txt":
		v_meas[0] = 1.02
		v_meas[1] = 0.92
		q_meas[0] = 0.605
		pij_meas[0] = 0.598
		qji_meas[0] = 0.305
	z = np.transpose([np.r_[v_meas, p_meas, q_meas, pij_meas, qij_meas, pji_meas, qji_meas]])

	n = len(ps.y_bus)*2 - 1
	m = len(z)  # number of measurements
	nbus = len(ps.bus_data[:, 0])
	nbr = len(ps.branch_data[:, 0])

	# ~~~~~ Creat covariance matrix, R and inverse, W ~~~~~
	r = np.zeros((m, m))
	for i in range(m):
		if i < nbus:
			r[i, i] = v_stdev**2
		elif i < 2*nbus:
			r[i, i] = p_stdev**2
		elif i < 3*nbus:
			r[i, i] = q_stdev**2
		elif i < 3*nbus + nbr:
			r[i, i] = pij_stdev**2
		elif i < 3*nbus + 2*nbr:
			r[i, i] = qij_stdev**2
		elif i < 3*nbus + 3*nbr:
			r[i, i] = pij_stdev**2
		elif i < 3*nbus + 4*nbr:
			r[i, i] = qij_stdev**2
	w = inv(r)

	# ~~~~~ Iterate State Estimator ~~~~~
	d_est = np.transpose([d0])
	v_est = np.transpose([v0])
	for it in range(40):
		h = ps.se_h_matrix(v_est.T[0], d_est.T[0])
		dz = z - h_calc(ps, v_est, d_est)
		gain = h.T @ w @ h
		dx = inv(gain) @ h.T @ w @ dz
		print("max dx = ", max(abs(dx)))
		if max(abs(dx)) < 0.005:
			print("iteration: ", it)
			break
		d_est[1:] = d_est[1:] + dx[0:len(d0)-1]  # -1 because first angle left out
		v_est = v_est + dx[len(d0)-1:]
	print(d - np.transpose(d_est)[0])
	print(v - np.transpose(v_est)[0])
	print(np.mean(d - np.transpose(d_est)[0]), np.mean(v - np.transpose(v_est)[0]))

	# ~~~~~ Bad data test (X^2 test) ~~~~~
	e_est = dz
	r_p = (r - h @ inv(gain) @ h.T)
	r_p_jj = (r_p * np.eye(m))
	j_wls = sum(w @ dz ** 2)
	exp_j = sum(sum(r_p_jj*w))
	print('exp_j=',exp_j)
	k = m - n  # degrees of freedom
	print("k=", k)
	alpha = 0.01  # 99% confidence interval
	from scipy.stats.distributions import chi2
	print(chi2.ppf(1-alpha, df=k))


	def chi_cdf(x, k):
		from scipy.special import gammainc
		return gammainc(k/2, x/2)

	print(1-chi_cdf(130, k))  # critical value is 65.9

	def ichi_cdf(x, k):
		from math import gamma
		return gamma(k/2, 1/(2*x))/gamma()

