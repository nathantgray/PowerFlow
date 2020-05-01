from dynamics import *
from KundurDynInit import *

if __name__ == "__main__":
	dps = DynamicSystem(case_name, udict, sparse=False)
	v0, d0 = dps.flat_start()
	v, d, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
	y = np.r_[d[1:], v[1:]]
	type3 = dps.type3_init()
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

	# Transient Stability Type 3

	x = np.array([])

	vgen = np.r_[dps.v[dps.slack], E3 * np.exp(1j * th3)]
	Igen = dps.y_gen.dot(vgen)
	Pe = (vgen * np.conj(Igen)).real[dps.pv]
	y_gen_pre = dps.y_gen.copy()
	x_pre = x3
	x = np.r_[x, x3]
	# Fault On
	fault_case = 'Kundur_modified_fault.txt'
	dpsf = DynamicSystem(fault_case, udict, sparse=False)
	def f(x_states):
		return dpsf.dyn3_f(x_states, E3, Pm3)
	dpsf.y_gen = dpsf.makeygen(v_override=dps.v)
	y_gen_fault = dpsf.makeygen(v_override=dps.v)
	clearing_cycles = 3  # critical clearing time is 22 cycles
	freq = 60
	t = 0
	tc = clearing_cycles/freq
	h = 0.001
	xk = x_pre.copy()
	while t < tc:
		t = t + h
		xk = xk + h * f(xk)
		x = np.c_[x, xk]

		# unpack x
		ng = len(dps.ws) + 1
		th = np.array([])
		w = np.array([])
		for i in range(ng - 1):
			j = i * 2
			th = np.r_[th, xk[0 + j]]
			w = np.r_[w, xk[1 + j]]
		vgen = np.r_[dpsf.v[dpsf.slack], E3 * np.exp(1j * th)]
		Igen = y_gen_fault.dot(vgen)
		Pe = np.c_[Pe, (vgen * np.conj(Igen)).real[dpsf.pv]]
	# Fault Cleared
	clear_case = 'Kundur_modified_cleared.txt'
	dpsc = DynamicSystem(clear_case, udict, sparse=False)
	def f(x_states):
		return dpsc.dyn3_f(x_states, E3, Pm3)
	dpsc.y_gen = dpsc.makeygen(v_override=dps.v)
	y_gen_clear = dpsc.makeygen(v_override=dps.v)
	t_end = 3
	while t < t_end:
		t = t + h
		xk = xk + h * f(xk)
		x = np.c_[x, xk]

		# unpack x
		ng = len(dps.ws) + 1
		th = np.array([])
		w = np.array([])
		for i in range(ng - 1):
			j = i * 2
			th = np.r_[th, xk[0 + j]]
			w = np.r_[w, xk[1 + j]]
		vgen = np.r_[dpsc.v[dpsc.slack], E3 * np.exp(1j * th)]
		Igen = y_gen_clear.dot(vgen)
		Pe = np.c_[Pe, (vgen * np.conj(Igen)).real[dpsc.pv]]

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.plot(x[0, :]*180/pi, Pe[0, :])
	ax1.plot(x[2, :]*180/pi, Pe[1, :])
	ax1.plot(x[4, :]*180/pi, Pe[2, :])
	ax1.set_ylabel('P (p.u.)')
	ax1.set_xlabel('Theta (degrees)')
	ax1.set_title('Type-3 P-Th Response - tc={}ms={} cycles, KD={}'.format(round(tc*1000), clearing_cycles, dps.Kd[0]))
	plt.show()
	# fig.savefig("p-th_type3_3cycle.png")
	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111)
	ax2.plot(x[0, :]*180/pi)
	ax2.plot(x[2, :]*180/pi)
	ax2.plot(x[4, :]*180/pi)
	ax2.set_ylabel('Theta (degrees)')
	ax2.set_xlabel('time (ms)')
	ax2.set_title('Type-3 Angle Response - tc={}ms={} cycles, KD={}'.format(round(tc*1000), clearing_cycles, dps.Kd[0]))
	plt.show()
	# fig2.savefig("theta_type3_3cycle.png")
	fig3 = plt.figure()
	ax3 = fig3.add_subplot(111)
	ax3.plot(Pe[0, :])
	ax3.plot(Pe[1, :])
	ax3.plot(Pe[2, :])
	ax3.set_ylabel('Power (p.u.)')
	ax3.set_xlabel('time (ms)')
	ax3.set_title('Type-3 Power Response - tc={}ms={} cycles, KD={}'.format(round(tc*1000), clearing_cycles, dps.Kd[0]))
	plt.show()
	# fig3.savefig("power_type3_3cycle.png")


