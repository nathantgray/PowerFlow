import numpy as np
from scipy.stats.distributions import chi2
from copy import deepcopy
from classes.power_system import PowerSystem
from numpy.linalg import inv


if __name__ == "__main__":
	case_name = "IEEE14BUS.txt"
	# case_name = "IEEE14BUS_handout.txt"
	# case_name = "2BUS.txt"
	ps = PowerSystem(case_name, sparse=False)
	v0, d0 = ps.flat_start()
	# ~~~~~ Create Measurements ~~~~~
	v, d, it = ps.pf_newtonraphson(v0, d0, prec=5, maxit=10, qlim=False)
	s = (v*np.exp(1j*d))*np.conj(ps.y_bus.dot(v*np.exp(1j*d)))
	p = np.real(s)
	q = np.imag(s)
	pij, qij, pji, qji = ps.branch_flows(v, d)
	v_stdev = 0.01
	p_stdev = 0.03
	q_stdev = 0.03
	pij_stdev = 0.02
	qij_stdev = 0.02
	np.random.seed(6)
	# Generate random noise.
	v_noise = np.random.normal(0, v_stdev, v.shape)
	p_noise = np.random.normal(0, p_stdev, p.shape)
	q_noise = np.random.normal(0, q_stdev, q.shape)
	pij_noise = np.random.normal(0, pij_stdev, pij.shape)
	qij_noise = np.random.normal(0, qij_stdev, qij.shape)
	pji_noise = np.random.normal(0, pij_stdev, pji.shape)
	qji_noise = np.random.normal(0, qij_stdev, qji.shape)
	# Add random noise.
	v_meas = v + v_noise
	p_meas = p + p_noise
	q_meas = q + q_noise
	pij_meas = pij + pij_noise
	pji_meas = pji + pji_noise
	qij_meas = qij + qij_noise
	qji_meas = qji + qji_noise
	# ~~~~~ add bad data ~~~~~
	v_meas[3] = v[3] + 7*v_stdev
	# p_meas[3] = p[3] + 8*p_stdev
	# qji_meas[3] = qji[3] + 7*qij_stdev
	p_meas[12] = p[12] + 8*p_stdev
	qji_meas[9] = qji[9] - 7*qij_stdev
	z = np.transpose([np.r_[v_meas, p_meas, q_meas, pij_meas, qij_meas, pji_meas, qji_meas]])

	n = ps.y_bus.shape[0]*2 - 1  # state variables
	m = len(z)  # number of measurements
	nbus = len(ps.bus_data[:, 0])
	nbr = len(ps.branch_data[:, 0])

	# ~~~~~ Creat covariance matrix, R, and inverse, W ~~~~~
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
	w_all = inv(r)
	# ~~~~~ Iterate SE with Bad Data Detected ~~~~~
	bad_data_mapped = []  # indexes of bad data (based on list of all possible measurements)
	bad_indexes = []  # indexes of bad data before the index is mapped to original
	d_est = np.transpose([d0])
	v_est = np.transpose([v0])
	for bd_it in range(10):
		w = np.delete(w_all, bad_data_mapped, 0)
		w = np.delete(w, bad_data_mapped, 1)
		m = len(z) - len(bad_data_mapped)  # number of measurements used
		# ~~~~~ Iterate State Estimator ~~~~~
		for it in range(40):
			h = ps.se_h_matrix(v_est.T[0], d_est.T[0])
			h = np.delete(h, bad_data_mapped, 0)
			dz = z - ps.h_calc(v_est, d_est)
			dz = np.delete(dz, bad_data_mapped, 0)
			gain = h.T @ w @ h
			dx = inv(gain) @ h.T @ w @ dz
			print("max dx = ", max(abs(dx)), 'Iteration:', it)
			if max(abs(dx)) < 0.0005:
				break
			d_est[1:] = d_est[1:] + dx[0:len(d0)-1]  # -1 because first angle left out
			v_est = v_est + dx[len(d0)-1:]
		print("iterations: ", it)

		# ~~~~~ Bad data test (X^2 test) ~~~~~
		r_p = (inv(w) - h @ inv(gain) @ h.T)
		r_p_jj = (r_p * np.eye(m))
		j_wls = sum(w @ (dz ** 2))[0]
		e_norm = np.array([abs(dz[i])/np.sqrt(r_p[i, i]) for i in range(m)])
		k = m - n  # degrees of freedom
		alpha = 0.01  # 99% confidence interval
		critical_value = chi2.ppf(1-alpha, df=k)
		print("k=", k)
		print('**** j_wls=', j_wls)
		print('**** critical value=', chi2.ppf(1-alpha, df=k))
		worst_index = np.argmax(e_norm)
		print('index=', worst_index)
		print('worst dz=', dz[worst_index])
		old_bad_indexes = deepcopy(bad_indexes)
		bad_indexes.append(worst_index)
		# ~~~~~ fix index to match full arrays ~~~~
		if j_wls > critical_value:
			if len(bad_data_mapped) > 0:
				for bad_index in reversed(old_bad_indexes):
					if worst_index >= bad_index:
						worst_index += 1
			bad_data_mapped.append(worst_index)
			print('Index of removed measurements:', bad_data_mapped)
		else:
			print('Success!')
			print('Index of removed measurements:', bad_data_mapped)
			break
