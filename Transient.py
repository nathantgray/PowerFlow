from dynamics import *
from KundurDynInit import *
from copy import deepcopy
if __name__ == "__main__":
	dps = DynamicSystem('Kundur_prefault.txt', udict, sparse=False, create_ygen=False)

	def g(y_input, x_states):
		return dps.dyn1_g(x_states, y_input)

	def f(x_states, y_input):
		return dps.dyn1_f(x_states, y_input, type1['vref'], type1['Pc'], limits=True)
	v0, d0 = dps.flat_start()
	v, d, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
	y = np.r_[d[1:], v[1:]]
	type1 = dps.type1_init()
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
	y_mis = dps.dyn1_g(x1, y)
	print('Type 1 x max error', np.max(np.abs(x_dot)))
	print('Type 1 y max error', np.max(np.abs(y_mis)))


	# Transient Stability
	# ==================================== Prefault ====================================
	x_pre = x1
	y_pre = y
	x = np.array([])
	x = np.r_[x, x_pre]

	# Calculate intermediate values
	vd = v[dps.pv] * sin(th1 - d[dps.pv])
	vq = v[dps.pv] * cos(th1 - d[dps.pv])
	Id = (Eqp1 - vq) / dps.xdp
	Iq = -(Edp1 - vd) / dps.xqp
	Pe = vd * Id + vq * Iq
	t_plot = 0
	# ==================================== Fault On  ====================================
	fault_case = 'Kundur_fault.txt'
	dpsf = DynamicSystem(fault_case, udict, sparse=False, create_ygen=False)
	y_bus_fault = dpsf.y_bus
	y_bus_pre = dps.y_bus
	dps.y_bus = y_bus_fault  # Change y_bus to faulted y_bus
	clearing_cycles = 0  # critical clearing time is 78 cycles
	freq = 60
	t = 0
	tc = clearing_cycles/freq
	h = 0.001
	xk = x_pre.copy()
	x = np.c_[x, xk]
	yf = fsolve(g, y_pre, args=(x_pre))
	# Calculate Pe for plotting
	n = len(yf) // 2 + 1
	ng = len(dps.ws) + 1
	th = np.array([])
	Eqp = np.array([])
	Edp = np.array([])
	for i in range(ng - 1):
		j = i * 6
		th = np.r_[th, xk[0 + j]]
		Eqp = np.r_[Eqp, xk[2 + j]]
		Edp = np.r_[Edp, xk[3 + j]]
	# unpack y
	d = np.r_[0, y[0:n - 1]]
	v = np.r_[dps.vslack, yf[n - 1:2 * n - 2]]
	# Calculate intermediate values
	vd = v[dps.pv] * sin(th - d[dps.pv])
	vq = v[dps.pv] * cos(th - d[dps.pv])
	Id = (Eqp - vq) / dps.xdp
	Iq = -(Edp - vd) / dps.xqp
	Pe = np.c_[Pe, vd * Id + vq * Iq]
	t_plot = np.r_[t_plot, t]
	print('y error', np.max(np.abs(g(yf, x_pre))))
	while t < tc:
		t = t + h
		t_plot = np.r_[t_plot, t]
		xk = xk + h * f(xk, yf)
		x = np.c_[x, xk]
		yf = fsolve(g, yf, args=(xk))
		print('y error', np.max(np.abs(g(yf, xk))))
		print(t)
		# Calculate Pe for plotting
		n = len(yf) // 2 + 1
		ng = len(dps.ws) + 1
		# unpack x
		th = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		for i in range(ng - 1):
			j = i * 6
			th = np.r_[th, xk[0 + j]]
			Eqp = np.r_[Eqp, xk[2 + j]]
			Edp = np.r_[Edp, xk[3 + j]]
		# unpack y
		d = np.r_[0, y[0:n - 1]]
		v = np.r_[dps.vslack, y[n - 1:2 * n - 2]]
		# Calculate intermediate values
		vd = v[dps.pv] * sin(th - d[dps.pv])
		vq = v[dps.pv] * cos(th - d[dps.pv])
		Id = (Eqp - vq) / dps.xdp
		Iq = -(Edp - vd) / dps.xqp
		Pe = np.c_[Pe, vd * Id + vq * Iq]
	print('Fault cleared')
	# ==================================== Fault Cleared =====================================
	clear_case = 'Kundur_clear.txt'
	dpsc = DynamicSystem(clear_case, udict, sparse=False, create_ygen=False)
	y_bus_clear = dpsc.y_bus
	dps.y_bus = y_bus_clear  # Change y_bus to cleared y_bus
	t_end = 2
	yf = fsolve(g, y_pre, args=(xk))
	while t < t_end:
		t = round(t + h, 4)
		t_plot = np.r_[t_plot, t]
		xk = xk + h * f(xk, yf)
		x = np.c_[x, xk]
		yf = fsolve(g, yf, args=(xk))
		print(t)
		# Calculate Pe for plotting
		n = len(yf) // 2 + 1
		ng = len(dps.ws) + 1
		# unpack x
		th = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		for i in range(ng - 1):
			j = i * 6
			th = np.r_[th, xk[0 + j]]
			Eqp = np.r_[Eqp, xk[2 + j]]
			Edp = np.r_[Edp, xk[3 + j]]
		# unpack y
		d = np.r_[0, yf[0:n - 1]]
		v = np.r_[dps.vslack, yf[n - 1:2 * n - 2]]
		# Calculate intermediate values
		vd = v[dps.pv] * sin(th - d[dps.pv])
		vq = v[dps.pv] * cos(th - d[dps.pv])
		Id = (Eqp - vq) / dps.xdp
		Iq = -(Edp - vd) / dps.xqp
		Pe = np.c_[Pe, vd * Id + vq * Iq]
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(x[0, :]*180/pi, Pe[0, :])
	ax1.plot(x[6, :]*180/pi, Pe[1, :])
	ax1.plot(x[12, :]*180/pi, Pe[2, :])
	ax1.set_ylabel('P (p.u.)')
	ax1.set_xlabel('Theta (degrees)')
	ax1.set_title('Type-1 P-Th Response - tc={}ms={} cycles, KD={}, R={}%'.format(round(tc*1000), clearing_cycles, dps.Kd[0], round(dps.R[0]*100, 2)))
	plt.show()
	# fig.savefig("p-th_type1_3cycle.png")
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	ax2.plot(x[0, :]*180/pi)
	ax2.plot(x[6, :]*180/pi)
	ax2.plot(x[12, :]*180/pi)
	ax2.set_ylabel('Theta (degrees)')
	ax2.set_xlabel('time (ms)')
	ax2.set_title('Type-1 Angle Response - tc={}ms={} cycles, KD={}, R={}%'.format(round(tc*1000), clearing_cycles, dps.Kd[0], round(dps.R[0]*100, 2)))
	fig2.show()
	# fig2.savefig("theta_type1_3cycle.png")
	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111)
	ax3.plot(Pe[0, :])
	ax3.plot(Pe[1, :])
	ax3.plot(Pe[2, :])
	ax3.set_ylabel('Power (p.u.)')
	ax3.set_xlabel('time (ms)')
	ax3.set_title('Type-1 Power Response - tc={}ms={} cycles, KD={}, R={}%'.format(round(tc*1000), clearing_cycles, dps.Kd[0], round(dps.R[0]*100, 2)))
	plt.show()
	# fig3.savefig("power_type1_3cycle.png")



