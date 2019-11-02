

if __name__ == "__main__":
	from power_system import PowerSystem
	import numpy as np
	import matplotlib.pyplot as plt
	case_name = "IEEE14BUS_handout.txt"
	ps = PowerSystem(case_name, sparse=True)
	v0, d0 = ps.flat_start()
	v, d, it = ps.pf_newtonraphson(v0, d0, prec=2, maxit=10, qlim=True)
	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	p = np.real(s)
	q = np.real(s)
	pij, qij, pji, qji = ps.branch_flows(v, d)
	v_stdev = 0.03
	p_stdev = 0.02
	q_stdev = 0.05
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
	print(pij)
	print(qij)
