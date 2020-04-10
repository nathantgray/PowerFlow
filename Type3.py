from dynamics import *


if __name__ == "__main__":
	case_name = 'Kundur_modified.txt'
	zbase_conv = 0.0008401596
	sbase_conv = 9
	ws = 2 * pi * 60
	H = 6.5 * sbase_conv
	Kd = 0
	Td0 = 8
	Tq0 = 0.4
	xd = 1.8 * zbase_conv
	xdp = 0.3 * zbase_conv
	xq = 1.7 * zbase_conv
	xqp = 0.55 * zbase_conv
	Rs = 0.0025 * zbase_conv

	H1 = 6.5 * sbase_conv
	H3 = 6.175 * sbase_conv

	# vref = ps.bus_data[ps.pv, ps.busDesiredVolts]
	# Pc = ps.bus_data[ps.pv, ps.busGenMW]/ps.p_base
	udict = {
		'ws': np.array([ws, ws, ws]),
		'H': np.array([H1, H3, H3]),
		'Kd': np.array([2, 2, 2]),
		'Td0': np.array([Td0, Td0, Td0]),
		'Tq0': np.array([Tq0, Tq0, Tq0]),
		'xd': np.array([xd, xd, xd]),
		'xdp': np.array([xdp, xdp, xdp]),
		'xq': np.array([xq, xq, xq]),
		'xqp': np.array([xqp, xqp, xqp]),
		'Rs': np.array([0, 0, 0]),
		'Ka': np.array([50, 50, 50]),
		'Ta': np.array([0.01, 0.01, 0.01]),
		'Vr_min': np.array([-4, -4, -4]),
		'Vr_max': np.array([4, 4, 4]),
		'Efd_min': np.array([0, 0, 0]),
		'Efd_max': np.array([2, 2, 2]),
		'Tsg': np.array([100, 100, 100]),
		'Ksg': np.array([1, 1, 1]),
		'Psg_min': np.array([0, 0, 0]),
		'Psg_max': np.array([1, 1, 1]),
		'R': np.array([0.05, 0.05, 0.05])
	}
	dps = DynamicSystem(case_name, udict, sparse=False)
	v0, d0 = dps.flat_start()
	v, d, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
	gens = np.r_[dps.slack, dps.pv]
	v_ = (v * np.exp(1j * d))  # voltage phasor
	s = v_ * np.conj(dps.y_bus.dot(v_))
	s_l = dps.p_load_full + 1j*dps.q_load_full
	s_g = s + s_l
	I_g = np.conj(s_g/(v_))
	E_pv = v_[dps.pv] + 1j*dps.xp*I_g[dps.pv]
	E = np.r_[v_[dps.slack], E_pv]  # only generator voltages
	E_mag = np.abs(E)
	Pm = s.real[dps.pv]
	y_gen = dps.makeygen()
	th0 = d[dps.pv]
	w0 = np.ones(len(dps.pv)) - 0.5
	x0 = np.array([])
	for i in range(len(dps.pv)):
		x0 = np.r_[x0, th0[i], w0[i]]
	xe = fsolve(dps.dyn3_f, x0, args=(E_mag, Pm))
	print(xe)
	# print(d_nr*180/pi)