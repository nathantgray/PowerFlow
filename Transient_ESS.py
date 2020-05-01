from dynamics import *
from KundurDynInit_ess import *
from copy import deepcopy
def p_calc(dps, x, y):
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
	vs_last = v[7]
	# Calculate intermediate values
	vd = v[dps.pv] * sin(th - d[dps.pv])
	vq = v[dps.pv] * cos(th - d[dps.pv])
	Id = (Eqp - vq) / dps.xdp
	Iq = -(Edp - vd) / dps.xqp
	return vd * Id + vq * Iq

if __name__ == "__main__":
	dps = DynamicSystem('Kundur_modified.txt', udict, sparse=False, create_ygen=False)

	def g(y_input, x_states):
		return dps.dyn1_g(x_states, y_input)

	def f(x_states, y_input):
		return dps.dyn1_f(x_states, y_input, type1['vref'], type1['Pc'], limits=False)

	def g_ess(y_input, x_states):
		return dps.dyn1ess_g(x_states, y_input)

	def f_ess(x_states, y_input, vs_dot, P_A_dot, Q_A_dot):
		return dps.dyn1ess_f(x_states, y_input, type1['vref'], type1['Pc'], vs_ref, vdc_ref, vs_dot, P_A_dot, Q_A_dot, limits=False)

	# ==================================== Initialize System ====================================
	v0, d0 = dps.flat_start()
	v, d, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
	vs_last = v[7]
	vs_ref = v[7]
	vdc_ref = vs_ref*1.2
	v_dc = vs_ref*1.2
	vs_dot = 0
	phi = 0
	m = v[7]/v_dc
	v_dc = vs_ref*1.2
	uw_m = 0
	u1_m = 0
	upss_m = 0
	uw_f = 0
	u1_f = 0
	upss_f = 0
	n_ess = 9
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
	x_pre = np.r_[x1, m, phi, v_dc, uw_m, u1_m, upss_m, uw_f, u1_f, upss_f]
	x_dot = dps.dyn1ess_f(x_pre, y, type1['vref'], type1['Pc'], vs_ref, vdc_ref, vs_dot, 0, 0, limits=False)
	y_mis = dps.dyn1ess_g(x_pre, y)
	print('Type 1 x max error', np.max(np.abs(x_dot)))
	print('Type 1 y max error', np.max(np.abs(y_mis)))
	# print(P_A)
	print((d[6] - d[8])*180/pi)
	# Transient Stability
	# ==================================== Initialize simulation output variables====================================
	P_A = np.array([dps.pij_flow(d, v, 6, 7), dps.pij_flow(d, v, 6, 7)])
	P_A_ess = np.array([dps.pij_flow(d, v, 6, 7), dps.pij_flow(d, v, 6, 7)])
	Q_A_ess = np.array([dps.qij_flow(d, v, 6, 7, 3), dps.qij_flow(d, v, 6, 7, 3)])

	y_pre = y.copy()
	x = x_pre[:18].copy()
	x_ess = x_pre.copy()
	# Calculate intermediate values
	vd = v[dps.pv] * sin(th1 - d[dps.pv])
	vq = v[dps.pv] * cos(th1 - d[dps.pv])
	Id = (Eqp1 - vq) / dps.xdp
	Iq = -(Edp1 - vd) / dps.xqp
	Pe = vd * Id + vq * Iq
	Pe_ess = vd * Id + vq * Iq
	vs = np.array([v[7], v[7]])
	Ps, Qs = dps.ess_out(x_ess[-n_ess:], vs[-1])
	t_plot = 0
	t = 0
	h = 0.001
	xk_ess = x_pre.copy()
	xk = x_pre[:18].copy()
	yf = y_pre.copy()
	yf_ess = yf.copy()
	n = len(yf) // 2 + 1
	freq = 60
	yf_ess = yf.copy()
	# ==================================== Simulate each 1 ms ====================================
	print('pre disturbance')
	t_end = 3
	while t < t_end:
		if t == 0.2:
			print('start disturbance')
		if 0.2 <= t < 0.25:
			xk[17] = Pm1[2]*(1.005)
			xk_ess[17] = Pm1[2]*(1.005)
		if t == 0.3:
			print('post disturbance')
			xk[17] = Pm1[2]
		t = round(t + h, 4)
		t_plot = np.r_[t_plot, t]
		#
		# xk = xk + h * f(xk, yf)
		# yf = fsolve(g, yf, args=(xk))
		# d = np.r_[0, yf[0:n - 1]]
		# v = np.r_[dps.vslack, yf[n - 1:2 * n - 2]]
		# x = np.c_[x, xk]
		# P_A = np.r_[P_A, dps.pij_flow(d, v, 6, 7)]
		# Pe = np.c_[Pe, p_calc(dps, xk, yf)]

		vs_dot = (vs[-1] - vs[-2]) / h
		P_A_dot = (P_A_ess[-1] - P_A_ess[-2])/h
		Q_A_dot = (Q_A_ess[-1] - Q_A_ess[-2])/h
		xk_ess = xk_ess + h * f_ess(xk_ess, yf_ess, vs_dot, P_A_dot, Q_A_dot)
		yf_ess = fsolve(g_ess, yf_ess, args=(xk_ess))

		x_ess = np.c_[x_ess, xk_ess]
		d_ess = np.r_[0, yf_ess[0:n - 1]]
		v_ess = np.r_[dps.vslack, yf_ess[n - 1:2 * n - 2]]
		vs = np.r_[vs, v_ess[7]]
		Ps = np.r_[Ps, dps.ess_out(xk_ess[-n_ess:], vs[-1])[0]]
		Qs = np.r_[Qs, dps.ess_out(xk_ess[-n_ess:], vs[-1])[1]]
		Pe_ess = np.c_[Pe_ess, p_calc(dps, xk_ess, yf_ess)]
		P_A_ess = np.r_[P_A_ess, dps.pij_flow(d_ess, v_ess, 6, 7)]
		Q_A_ess = np.r_[Q_A_ess, dps.qij_flow(d_ess, v_ess, 6, 7, 3)]
	# ==================================== Plot ====================================
	# fig3 = plt.figure()
	# ax3 = fig3.add_subplot(111)
	# ax3.plot(Pe[0, :])
	# ax3.plot(Pe[1, :])
	# ax3.plot(Pe[2, :])
	# ax3.set_ylabel('Power (p.u.)')
	# ax3.set_xlabel('time (ms)')
	# ax3.set_title('Type-1 Power Response - KD={}, R={}%'.format(dps.Kd[0], round(dps.R[0]*100, 2)))
	# plt.show()
	# fig3.savefig("power_type1_3cycle.png")
	fig4 = plt.figure()
	ax4 = fig4.add_subplot(111)
	# ax4.plot(P_A)
	ax4.plot(P_A_ess, '.')
	ax4.plot(Ps, '.')
	ax4.set_ylabel('Power (p.u.)')
	ax4.set_xlabel('time (ms)')
	ax4.set_title('Tie line Power Response - KD={}, R={}%'.format(dps.Kd[0], round(dps.R[0]*100, 2)))
	plt.show()
	# fig3.savefig("power_type1_3cycle.png")
	fig4 = plt.figure()
	ax4 = fig4.add_subplot(111)
	ax4.plot(vs)
	ax4.set_ylabel('Power (p.u.)')
	ax4.set_xlabel('time (ms)')
	ax4.set_title('Tie line Voltage Response')
	plt.show()




