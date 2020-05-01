from dynamics import *
from KundurDynInit import *

if __name__ == "__main__":
	dps = DynamicSystem(case_name, udict, sparse=False)
	v0, d0 = dps.flat_start()
	v, d, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
	y = np.r_[d[1:], v[1:]]
	type2 = dps.type2_init()
	# Check type 2
	th = type2['th']
	w = np.ones(len(dps.pv))
	Eqp = type2['Eqp']
	Edp = type2['Edp']
	va = type2['Efd']
	Pm = type2['Pg']
	x2 = np.array([])
	for i in range(len(dps.pv)):
		x2 = np.r_[x2, th[i], w[i], Eqp[i], Edp[i], va[i], Pm[i]]
	x_dot = dps.dyn2_f(x2, type2['vref'], type2['Pc'])
	print('Type 2 max error', np.max(np.abs(x_dot)))
	# print(x_dot)

	# Transient Stability Type 2

	x = np.array([])
	# Prefault
	vgen = np.r_[dps.v[dps.slack], (Edp + 1j * Eqp) * (1 * np.exp(1j * (th - pi / 2)))]
	Igen = dps.y_gen.dot(vgen)
	v = np.abs(vgen[1:] - Igen[1:] * 1j * dps.xp)
	Pe = (vgen * np.conj(Igen)).real[dps.pv]
	y_gen_pre = dps.y_gen.copy()
	x_pre = x2
	x = np.r_[x, x2]

	# Fault On
	fault_case = 'Kundur_modified_fault.txt'
	dpsf = DynamicSystem(fault_case, udict, sparse=False)
	dpsf.y_gen = dpsf.makeygen(v_override=dps.v)
	y_gen_fault = dpsf.makeygen(v_override=dps.v)
	def f(x_input):
		return dpsf.dyn2_f(x_input, type2['vref'], type2['Pc'], limits=True)
	clearing_cycles = 3  # critical clearing time is 33 cycles
	freq = 60
	t = 0
	tc = clearing_cycles/freq
	h = 0.001
	xk = x_pre.copy()
	while t < tc:
		t = t + h
		xk = xk + h * f(xk)
		x = np.c_[x, xk]
		# Calculate Pe for plotting.
		# unpack x
		ng = len(dps.ws) + 1
		th = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		for i in range(ng - 1):
			j = i * 6
			th = np.r_[th, xk[0 + j]]
			Eqp = np.r_[Eqp, xk[2 + j]]
			Edp = np.r_[Edp, xk[3 + j]]
		vgen = np.r_[dpsf.v[dpsf.slack], (Edp + 1j * Eqp) * (1 * np.exp(1j * (th - pi / 2)))]
		Igen = y_gen_fault.dot(vgen)
		v = np.abs(vgen[1:] - Igen[1:] * 1j * dpsf.xp)
		Id = (Igen[dpsf.pv] * np.exp(1j * (pi / 2 - th))).real
		Iq = (Igen[dpsf.pv] * np.exp(1j * (pi / 2 - th))).imag
		Pe = np.c_[Pe, (vgen * np.conj(Igen)).real[dpsf.pv]]
	# Fault Cleared
	clear_case = 'Kundur_modified_cleared.txt'
	dpsc = DynamicSystem(clear_case, udict, sparse=False)
	dpsc.y_gen = dpsc.makeygen(v_override=dps.v)
	y_gen_clear = dpsc.makeygen(v_override=dps.v)
	def f(x_input):
		return dpsc.dyn2_f(x_input, type2['vref'], type2['Pc'], limits=True)
	t_end = 20
	while t < t_end:
		t = t + h
		xk = xk + h * f(xk)
		x = np.c_[x, xk]

		# unpack x
		th = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		for i in range(ng - 1):
			j = i * 6
			th = np.r_[th, xk[0 + j]]
			Eqp = np.r_[Eqp, xk[2 + j]]
			Edp = np.r_[Edp, xk[3 + j]]
		vgen = np.r_[dpsc.v[dpsc.slack], (Edp + 1j * Eqp) * (1 * np.exp(1j * (th - pi / 2)))]
		Igen = y_gen_clear.dot(vgen)
		v = np.abs(vgen[1:] - Igen[1:] * 1j * dpsc.xp)
		Id = (Igen[dpsc.pv] * np.exp(1j * (pi / 2 - th))).real
		Iq = (Igen[dpsc.pv] * np.exp(1j * (pi / 2 - th))).imag
		Pe = np.c_[Pe, (vgen * np.conj(Igen)).real[dpsc.pv]]
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(x[0, :]*180/pi, Pe[0, :])
	ax1.plot(x[6, :]*180/pi, Pe[1, :])
	ax1.plot(x[12, :]*180/pi, Pe[2, :])
	ax1.set_ylabel('P (p.u.)')
	ax1.set_xlabel('Theta (degrees)')
	ax1.set_title('Type-2 P-Th Response - tc={}ms={} cycles, KD={}, R={}%'.format(round(tc*1000), clearing_cycles, dps.Kd[0], round(dps.R[0]*100, 2)))
	plt.show()
	# fig.savefig("p-th_type2_3cycle.png")
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	ax2.plot(x[0, :]*180/pi)
	ax2.plot(x[6, :]*180/pi)
	ax2.plot(x[12, :]*180/pi)
	ax2.set_ylabel('Theta (degrees)')
	ax2.set_xlabel('time (ms)')
	ax2.set_title('Type-2 Angle Response - tc={}ms={} cycles, KD={}, R={}%'.format(round(tc*1000), clearing_cycles, dps.Kd[0], round(dps.R[0]*100, 2)))
	fig2.show()
	# fig2.savefig("theta_type2_2cycle.png")
	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111)
	ax3.plot(Pe[0, :])
	ax3.plot(Pe[1, :])
	ax3.plot(Pe[2, :])
	ax3.set_ylabel('Power (p.u.)')
	ax3.set_xlabel('time (ms)')
	ax3.set_title('Type-2 Power Response - tc={}ms={} cycles, KD={}, R={}%'.format(round(tc*1000), clearing_cycles, dps.Kd[0], round(dps.R[0]*100, 2)))
	plt.show()
	# fig3.savefig("power_type2_3cycle.png")




