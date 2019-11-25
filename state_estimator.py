import numpy as np
from scipy.stats.distributions import chi2
from copy import deepcopy
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
	p_stdev = 0.02
	q_stdev = 0.02
	pij_stdev = 0.02
	qij_stdev = 0.02
	np.random.seed(5)
	v_noise = np.random.normal(0, v_stdev, v.shape)
	p_noise = np.random.normal(0, p_stdev, p.shape)
	q_noise = np.random.normal(0, q_stdev, q.shape)
	pij_noise = np.random.normal(0, pij_stdev, pij.shape)
	qij_noise = np.random.normal(0, qij_stdev, qij.shape)
	pji_noise = np.random.normal(0, pij_stdev, pji.shape)
	qji_noise = np.random.normal(0, qij_stdev, qji.shape)
	print('qji_std=', np.std(qji_noise))
	v_meas = v + v_noise
	p_meas = p + p_noise
	q_meas = q + q_noise
	pij_meas = pij + pij_noise
	pji_meas = pji + pji_noise
	qij_meas = qij + qij_noise
	qji_meas = qji + qji_noise
	# ~~~~~ add bad data ~~~~~
	# v_meas[0] = v[0] + 20*v_stdev
	# v_meas[1] = v[1] + -9*v_stdev
	# v_meas[2] = v[2] + 6*v_stdev
	# v_meas[3] = v[3] + 1000*v_stdev
	# v_meas[4] = v[4] + 4*v_stdev
	# pij_meas[1] = pij[1] - 3.1*pij_stdev
	# qji_meas[4] = qji[4] - 100.5*qij_stdev
	# qji_meas[1] = qji[1] - 100.5*qij_stdev
	# Test with 2 bus system from Grainger book example 15.6
	# if case_name == "2BUS.txt":
	# 	v_meas[0] = 1.02
	# 	v_meas[1] = 0.92
	# 	q_meas[0] = 0.605
	# 	pij_meas[0] = 0.598
	# 	qji_meas[0] = 0.305
	z = np.transpose([np.r_[v_meas, p_meas, q_meas, pij_meas, qij_meas, pji_meas, qji_meas]])

	n = len(ps.y_bus)*2 - 1  # state variables
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
	w_all = inv(r)
	# ~~~~~ Iterate SE with Bad Data Detected ~~~~~
	bad_data_mapped = []  # indexes of bad data (based on list of all possible measurements)
	bad_indexes =[]  # indexes of bad data before the index is mapped to original
	for bd_it in range(10):
		w = np.delete(w_all, bad_data_mapped, 0)
		w = np.delete(w, bad_data_mapped, 1)
		m = len(z) - len(bad_data_mapped)  # number of measurements used
		# ~~~~~ Iterate State Estimator ~~~~~
		d_est = np.transpose([d0])
		v_est = np.transpose([v0])
		for it in range(40):
			h = ps.se_h_matrix(v_est.T[0], d_est.T[0])
			h = np.delete(h, bad_data_mapped, 0)
			dz = z - h_calc(ps, v_est, d_est)
			dz = np.delete(dz, bad_data_mapped, 0)
			gain = h.T @ w @ h
			dx = inv(gain) @ h.T @ w @ dz
			# print("max dx = ", max(abs(dx)))
			if max(abs(dx)) < 0.0001:
				print("iteration: ", it)
				break
			d_est[1:] = d_est[1:] + dx[0:len(d0)-1]  # -1 because first angle left out
			v_est = v_est + dx[len(d0)-1:]
		# print(d - np.transpose(d_est)[0])
		# print(v - np.transpose(v_est)[0])
		# print(np.mean(d - np.transpose(d_est)[0]), np.mean(v - np.transpose(v_est)[0]))

		# ~~~~~ Bad data test (X^2 test) ~~~~~
		e_est = dz
		r_p = (inv(w) - h @ inv(gain) @ h.T)
		r_p_jj = (r_p * np.eye(m))
		j_wls = sum(w @ (dz ** 2))[0]
		e_norm = [abs(dz[i])/np.sqrt(r_p[i, i]) for i in range(m)]
		print('~~~~~~~ j_wls=', j_wls)
		k = m - n  # degrees of freedom
		print("k=", k)
		alpha = 0.01  # 99% confidence interval
		critical_value = chi2.ppf(1-alpha, df=k)
		print('~~~~~~~ critical value=', chi2.ppf(1-alpha, df=k))
		print('max abs(dz)=', max(abs(dz)))
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
			bad_data_mapped.append(worst_index)  # TODO fix indexing. This happens -> bad_data = [2, 0, 0]
			print(bad_data_mapped)
		else:
			print('Success!')
			print(bad_data_mapped)
			break


