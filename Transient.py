from dynamics import *


if __name__ == "__main__":
	case_name = 'Kundur_modified.txt'
	zbase_conv = 1/9
	sbase_conv = 9
	ws = 2 * pi * 60
	H = 6.5 * sbase_conv
	Kd = 2
	Td0 = 8
	Tq0 = 0.4
	xd = 1.8 * zbase_conv
	xdp = 0.3 * zbase_conv
	xq = 1.7 * zbase_conv
	xqp = 0.55 * zbase_conv
	Rs = 0  # 0.0025 * zbase_conv

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
		'Rs': np.array([Rs, Rs, Rs]),
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
		'R': np.array([0.05, 0.05, 0.05]),
		'Tw': np.array([1, 1, 1]),
		'K1': np.array([1, 1, 1]),
		'T1': np.array([0.1, 0.1, 0.1]),
		'T2': np.array([0.1, 0.1, 0.1]),
		'constZ': 0.5,
		'constP': 0.5,
		'constI': 0
	}
	dps = DynamicSystem(case_name, udict, sparse=False)
	v0, d0 = dps.flat_start()
	v, d, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
	y = np.r_[d[1:], v[1:]]
	type1 = dps.type1_init()
	type2 = dps.type2_init()
	type3 = dps.type3_init()
	# Check type 1
	th1 = type1['th']
	w1 = np.ones(len(dps.pv))
	Eqp1 = type1['Eqp']
	Edp1 = type1['Edp']
	va1 = type1['Efd']
	Pm1 = type1['Pg']
	x1 = np.array([])
	for i in range(len(dps.pv)):
		x1 = np.r_[x1, th1[i], w1[i], Eqp1[i], Edp1[i], va1[i], Pm1[i]]
	x_dot = dps.dyn1_f(x1, y, type1['vref'], type1['Pc'])
	print('Type 1 max error', np.max(np.abs(x_dot)))
	# Check type 2
	th2 = type2['th']
	w2 = np.ones(len(dps.pv))
	Eqp2 = type2['Eqp']
	Edp2 = type2['Edp']
	va2 = type2['Efd']
	Pm2 = type2['Pg']
	x2 = np.array([])
	for i in range(len(dps.pv)):
		x2 = np.r_[x2, th2[i], w2[i], Eqp2[i], Edp2[i], va2[i], Pm2[i]]
	x_dot = dps.dyn2_f(x2, type2['vref'], type2['Pc'])
	print('Type 2 max error', np.max(np.abs(x_dot)))
	# print(x_dot)

	# Transient Stability
	# Prefault
	x_pre = x1
	y_pre = y
	# Fault On
	fault_case = 'Kundur_modified_fault.txt'
	dpsf = DynamicSystem(fault_case, udict, sparse=False)
	shuntG = dpsf.bus_data[:, dpsf.busShuntG].copy()
	shuntB = dpsf.bus_data[:, dpsf.busShuntB].copy()
	y_bus_fault = dpsf.y_bus.copy()
	y_bus_pre = dps.y_bus.copy()
	v0 = v
	d0 = d
	step = 100
	print(np.max(np.abs(dpsf.dyn1_g(x1, y))))
	for i in range(step+1):
		# dpsf.bus_data[:, dpsf.busShuntG] = shuntG/step*i
		# dpsf.bus_data[:, dpsf.busShuntB] = shuntB/step*i
		# dpsf.y_bus = dpsf.makeybus()
		dpsf.y_bus = y_bus_pre + i/step*(y_bus_fault - y_bus_pre)
		vf, df, it = dpsf.pf_newtonraphson(v0, d0, prec=3, maxit=30, qlim=False, verbose=False)
		yf = np.r_[df[1:], vf[1:]]
		init_fault = dpsf.type1_init()
		# Check type 1
		th1 = init_fault['th']
		w1 = np.ones(len(dps.pv))
		Eqp1 = init_fault['Eqp']
		Edp1 = init_fault['Edp']
		va1 = init_fault['Efd']
		Pm1 = init_fault['Pg']
		x1 = np.array([])
		for i in range(len(dps.pv)):
			x1 = np.r_[x1, th1[i], w1[i], Eqp1[i], Edp1[i], va1[i], Pm1[i]]
		x_dot = dpsf.dyn1_f(x1, yf, init_fault['vref'], init_fault['Pc'])
		print('Type 1 max error', np.max(np.abs(x_dot)))
		# print(np.max(np.abs(dpsf.dyn1_g(x1, np.r_[df[1:], vf[1:]]))))
		if it == 30:
			print('failed', i/step)
			break
		v0 = vf
		d0 = df
		print(vf, it, i)




