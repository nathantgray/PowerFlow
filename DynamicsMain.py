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
	print(x_dot)
	# Check type 3
	th3 = type3['th']
	w3 = np.ones(len(dps.pv))
	Pm3 = type3['Pg']
	E3 = type3['E']
	x3 = np.array([])
	for i in range(len(dps.pv)):
		x3 = np.r_[x3, th3[i], w3[i]]
	x_dot = dps.dyn3_f(x3, E3, Pm3)
	print('Type 3 max error', np.max(np.abs(x_dot)))
	print(x_dot)

	J1 = dps.J_dyn1(x1, y, type1['vref'], type1['Pc'])
	ev1, vl1, vr1 = eig(J1, left = True)
	w1 = inv(vr1)
	J2 = dps.J_dyn2(x2, type2['vref'], type2['Pc'])
	ev2, vl2, vr2 = eig(J2, left = True)
	w2 = inv(vr2)
	J3 = dps.J_dyn3(x3, E3, Pm3)
	ev3, vl3, vr3 = eig(J3, left = True)
	w3 = inv(vr3)

	threshold = 0.01
	states1 = ['th2', 'w2', 'Eqp2', 'Edp2', 'Efd2', 'Pm2',
			  'th3', 'w3', 'Eqp3', 'Edp3', 'Efd3', 'Pm3',
			  'th4', 'w4', 'Eqp4', 'Edp4', 'Efd4', 'Pm4']
	states2 = ['th2', 'w2', 'Eqp2', 'Edp2', 'Efd2', 'Pm2',
			  'th3', 'w3', 'Eqp3', 'Edp3', 'Efd3', 'Pm3',
			  'th4', 'w4', 'Eqp4', 'Edp4', 'Efd4', 'Pm4']
	states3 = ['th2', 'w2',
			  'th3', 'w3',
			  'th4', 'w4',]
	p_f1 = np.zeros_like(J1)
	print('\nType 1')
	for k in range(len(x1)):
		print('\n***Mode:', k)
		for i in range(len(x1)):
			p_f1[i, k] = np.abs(vr1[i, k]) * np.abs(w1[k, i]) / np.max(np.abs(vr1[:, k] * w1[k, :]))
			if p_f1[i, k] > threshold:
				print(states1[i], ':', round(p_f1[i, k], 2))

	print('\n-----Type 2-----')
	p_f2 = np.zeros_like(J2)
	for k in range(len(x2)):
		print('\n***Mode:', k)
		for i in range(len(x2)):
			p_f2[i, k] = np.abs(vr2[i, k]) * np.abs(w2[k, i]) / np.max(np.abs(vr2[:, k] * w2[k, :]))
			if p_f2[i, k] > threshold:
				print(states2[i], ':', round(p_f2[i, k], 2))

	print('\n-----Type 3-----')
	p_f3 = np.zeros_like(J3)
	for k in range(len(x3)):
		print('\n***Mode:', k)
		for i in range(len(x3)):
			p_f3[i, k] = np.abs(vr3[i, k]) * np.abs(w3[k, i]) / np.max(np.abs(vr3[:, k] * w3[k, :]))
			if p_f3[i, k] > threshold:
				print(states3[i], ':', round(p_f3[i, k], 2))

	f1 = ev1.imag/(2*pi)
	f2 = ev2.imag/(2*pi)
	f3 = ev3.imag/(2*pi)
	zeta1 = ev1.real/np.abs(ev1)
	zeta2 = ev2.real/np.abs(ev2)
	zeta3 = ev3.real/np.abs(ev3)
	damp1 = zeta1 * -100
	damp2 = zeta2 * -100
	damp3 = zeta3 * -100
	# print(ev)
	plt.plot(ev1.real, ev1.imag, '.')
	plt.grid(True, which='both')
	plt.show()
	plt.plot(ev2.real, ev2.imag, '.')
	plt.grid(True, which='both')
	plt.show()
	plt.plot(ev3.real, ev3.imag, '.')
	plt.grid(True, which='both')
	plt.show()

	fig = plt.figure()
	ax = fig.add_subplot(111, polar=True)
	ax.set_thetamin(0)
	ax.set_thetamax(90)
	angle = pi/2 - th1[0]
	# print('ref angle: ', angle*180/pi)
	plt.polar([angle, angle], [0, 1], label="ref")

	angle = pi/2 - (th1[0] - d[1])
	# print('v angle: ', angle*180/pi)
	plt.polar([angle, angle], [0, v[1]], label="v")
	angle = np.angle(Edp1[0] + 1j*Eqp1[0])
	# print('E angle: ', angle*180/pi)
	mag = np.abs(Edp1[0] + 1j*Eqp1[0])
	plt.polar([angle, angle], [0, mag], label="E")


	ax.legend()

	plt.show()

	print('end')

