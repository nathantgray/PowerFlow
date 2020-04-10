def dynamic_eqns(x, y, df, ps, bus):
	ibus = bus - 1
	u = df.iloc[:, bus]
	ws = u[0]
	H = u[1]
	Kd = u[2]
	Td0 = u[3]
	Tq0 = u[4]
	xd = u[5]
	xdp = u[6]
	xq = u[7]
	xqp = u[8]
	Rs = u[9]

	th = x[0]
	w = x[1]
	Eqp = x[2]
	Edp = x[3]
	Iq = x[4]
	Id = x[5]
	Pm = x[6]
	Efd = x[7]

	d = y[0][bus]
	v = y[1][bus]

	s = ps.complex_injections(y[1], y[0])
	Pg = s.real[bus]
	Qg = s.imag[bus]

	vd = v * sin(th - d)
	vq = v * cos(th - d)
	th_dot = (w - 1) * ws
	w_dot = 1 / (2 * H) * (Pm - Pg - Kd * (w - 1))  # Pg = vd * Id + vq * Iq
	Eqp_dot = 1 / Td0 * (-Eqp - (xd - xdp) * Id + Efd)
	Edp_dot = 1 / Tq0 * (-Edp + (xq - xqp) * Iq)
	dEqp = vq + Rs * Iq + xdp * Id - Eqp  # make rs = 0
	dEdp = vd + Rs * Id - xqp * Iq - Edp
	dPg = vd * Id + vq * Iq - Pg
	dQg = vq * Id - vd * Iq - Qg
	return [th_dot, w_dot, Eqp_dot, Edp_dot, dEqp, dEdp, dPg, dQg]


def dyn1_f(x, y, params, ps):
	n = len(y)//2 + 1
	u = params
	ws = u['ws']
	H = u['H']
	Kd = u['Kd']
	Td0 = u['Td0']
	Tq0 = u['Tq0']
	xd = u['xd']
	xdp = u['xdp']
	xq = u['xq']
	xqp = u['xqp']
	Ka = u['Ka']
	Ta = u['Ta']
	# Vr_min = u['Vr_min']
	# Vr_max = u['Vr_max']
	# Efd_min = u['Efd_min']
	# Efd_max = u['Efd_max']
	Tsg = u['Tsg']
	Ksg = u['Ksg']
	# Psg_min = u['Psg_min']
	# Psg_max = u['Psg_max']
	R = u['R']
	vref = u['vref']
	Pc = u['Pc']
	ng = len(ws) + 1
	# unpack x
	th = np.array([])
	w = np.array([])
	Eqp = np.array([])
	Edp = np.array([])
	va = np.array([])
	Pm = np.array([])
	# Efd = np.array([])
	# n_eq = len(x)/(ng - 1)
	for i in range(ng - 1):
		# j = i*n_eq
		j = i*6
		th = np.r_[th, x[0 + j]]
		w = np.r_[w, x[1 + j]]
		Eqp = np.r_[Eqp, x[2 + j]]
		Edp = np.r_[Edp, x[3 + j]]
		va = np.r_[va, x[4 + j]]
		Pm = np.r_[Pm, x[5 + j]]
		# Efd = np.r_[Efd, x[6 + j]] # should not be in the x vector
	Efd = va  # no limit equations
	# unpack y
	d = y[0:n - 1]
	v = y[n - 1:2*n - 2]

	# Calculate intermediate values
	i_pv = ps.pv - 1
	vd = v[i_pv] * sin(th - d[i_pv])
	vq = v[i_pv] * cos(th - d[i_pv])
	Id = (Eqp - vq) / xdp
	Iq = (Edp - vd) / xqp
	Pg = vd * Id + vq * Iq
	Qg = vq * Id - vd * Iq

	# calculate x_dot
	th_dot = (w - 1) * ws
	w_dot = 1 / (2 * H) * (Pm - Pg - Kd * (w - 1))  # Pg = vd * Id + vq * Iq
	Eqp_dot = 1 / Td0 * (-Eqp - (xd - xdp) * Id + Efd)
	Edp_dot = 1 / Tq0 * (-Edp + (xq - xqp) * Iq)
	va_dot = 1/Ta * (-Efd + Ka*(vref - va))
	Pm_dot = 1/Tsg*(-Pm + Ksg*(Pc + 1/R*(ws - w)))
	x_dot = np.array([])
	for i in range(ng - 1):
		x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i]]

	# va_dot = np.zeros((1, 3))
	# Pm_dot = np.zeros((1, 3))
	# for i in range(ng - 1):
	# 	if va[i] == Vr_max[i] and va_dot_unlim[i] > 0:
	# 		va_dot[i] = 0
	# 	elif va[i] == Vr_min[i] and va_dot_unlim[i] < 0:
	# 		va_dot[i] = 0
	# 	else:
	# 		va_dot[i] = va_dot_unlim
	#
	# 	if va[i] > Efd_max:
	# 		Efd[i] = Efd_max
	# 	elif va[i] < Efd_min:
	# 		Efd[i] = Efd_min
	# 	else:
	# 		Efd[i] = va[i]
	#
	# 	if Pm[i] == Psg_min[i] and Psg_dot_unlim[i] < 0:
	# 		Pm_dot[i] = 0
	# 	if Psg[i] == Psg_max[i] and Psg_dot_unlim[i] > 0:
	# 		Pm_dot[i] = 0
	# 	else:
	# 		Pm_dot[i] = Psg_dot_unlim[i]

	return x_dot


def dyn_jac(x, y, df, ps, bus):  # h = 10^-5
	ibus = bus - 1
	u = df.iloc[:, ibus]
	ws = u[0]
	H = u[1]
	Kd = u[2]
	Td0 = u[3]
	Tq0 = u[4]
	xd = u[5]
	xdp = u[6]
	xq = u[7]
	xqp = u[8]
	Rs = u[9]

	th = x[0]
	Iq = x[4]
	Id = x[5]

	d = y[0][ibus]
	v = y[1][ibus]

	jac = np.zeros((8, len(x)))

	jac[0, 1] = ws

	jac[1, 1] = -Kd / (2 * H)
	jac[1, 6] = 1 / (2 * H)

	jac[2, 2] = -1 / Td0
	jac[2, 5] = -1 / Td0 * (xd - xdp)
	jac[2, 7] = 1 / Td0

	jac[3, 3] = -1 / Tq0
	jac[3, 4] = 1 / Tq0 * (xq - xqp)

	jac[4, 0] = -v * sin(th - d)
	jac[4, 2] = -1
	jac[4, 4] = Rs
	jac[4, 5] = xdp

	jac[5, 0] = v * cos(th - d)
	jac[5, 3] = -1
	jac[5, 5] = Rs
	jac[5, 4] = xqp

	jac[6, 0] = v * cos(th - d) * Id - v * sin(th - d) * Iq
	jac[6, 5] = v * sin(th - d)
	jac[6, 4] = v * cos(th - d)

	jac[7, 0] = -v * sin(th - d) * Id - v * cos(th - d) * Iq
	jac[7, 5] = v * cos(th - d)
	jac[7, 4] = -v * sin(th - d)
	return jac


def dyn_A(x, y, df):
	Ng = 4  # Todo: get this number automatically
	ws = df.iloc[0, :]
	H = df.iloc[1, :]
	Kd = df.iloc[2, :]
	Td0 = df.iloc[3, :]
	Tq0 = df.iloc[4, :]
	xd = df.iloc[5, :]
	xdp = df.iloc[6, :]
	xq = df.iloc[7, :]
	xqp = df.iloc[8, :]
	Rs = df.iloc[9, :]

	th = [x[8 * i + 0] for i in range(Ng - 1)]
	w = [x[8 * i + 1] for i in range(Ng - 1)]
	Eqp = [x[8 * i + 2] for i in range(Ng - 1)]
	Edp = [x[8 * i + 3] for i in range(Ng - 1)]
	Iq = [x[8 * i + 4] for i in range(Ng - 1)]
	Id = [x[8 * i + 5] for i in range(Ng - 1)]
	Pm = [x[8 * i + 6] for i in range(Ng - 1)]
	Efd = [x[8 * i + 7] for i in range(Ng - 1)]

	d = y[0]
	v = y[1]

	jac = np.zeros((len(x), len(x)))
	for i in range(Ng - 1):
		ix = 8 * i
		ibus = i + 1
		jac[ix + 0, ix + 1] = ws[ibus]

		jac[ix + 1, ix + 1] = -Kd[ibus] / (2 * H[ibus])
		jac[ix + 1, ix + 6] = 1 / (2 * H[ibus])

		jac[ix + 2, ix + 2] = -1 / Td0[ibus]
		jac[ix + 2, ix + 5] = -1 / Td0[ibus] * (xd[ibus] - xdp[ibus])
		jac[ix + 2, ix + 7] = 1 / Td0[ibus]

		jac[ix + 3, ix + 3] = -1 / Tq0[ibus]
		jac[ix + 3, ix + 4] = 1 / Tq0[ibus] * (xq[ibus] - xqp[ibus])

		jac[ix + 4, ix + 0] = -v[ibus] * sin(th[i] - d[ibus])
		jac[ix + 4, ix + 2] = -1
		jac[ix + 4, ix + 4] = Rs[ibus]
		jac[ix + 4, ix + 5] = xdp[ibus]

		jac[ix + 5, ix + 0] = v[ibus] * cos(th[i] - d[ibus])
		jac[ix + 5, ix + 3] = -1
		jac[ix + 5, ix + 5] = Rs[ibus]
		jac[ix + 5, ix + 4] = xqp[ibus]

		jac[ix + 6, ix + 0] = v[ibus] * cos(th[i] - d[ibus]) * Id[i] - v[ibus] * sin(th[i] - d[ibus]) * Iq[i]
		jac[ix + 6, ix + 5] = v[ibus] * sin(th[i] - d[ibus])
		jac[ix + 6, ix + 4] = v[ibus] * cos(th[i] - d[ibus])

		jac[ix + 7, ix + 0] = -v[ibus] * sin(th[i] - d[ibus]) * Id[i] - v[ibus] * cos(th[i] - d[ibus]) * Iq[i]
		jac[ix + 7, ix + 5] = v[ibus] * cos(th[i] - d[ibus])
		jac[ix + 7, ix + 4] = -v[ibus] * sin(th[i] - d[ibus])
	return jac


def dyn_B(x, y, df, ps):
	Ng = 4  # TODO: get this number automatically
	ws = df.iloc[0, :]
	H = df.iloc[1, :]
	Kd = df.iloc[2, :]
	Td0 = df.iloc[3, :]
	Tq0 = df.iloc[4, :]
	xd = df.iloc[5, :]
	xdp = df.iloc[6, :]
	xq = df.iloc[7, :]
	xqp = df.iloc[8, :]
	Rs = df.iloc[9, :]

	th = [x[8 * i + 0] for i in range(Ng - 1)]
	w = [x[8 * i + 1] for i in range(Ng - 1)]
	Eqp = [x[8 * i + 2] for i in range(Ng - 1)]
	Edp = [x[8 * i + 3] for i in range(Ng - 1)]
	Iq = [x[8 * i + 4] for i in range(Ng - 1)]
	Id = [x[8 * i + 5] for i in range(Ng - 1)]
	Pm = [x[8 * i + 6] for i in range(Ng - 1)]
	Efd = [x[8 * i + 7] for i in range(Ng - 1)]

	d = y[0]
	v = y[1]

	N = len(d)
	jac = np.zeros((len(x), 2 * (N - 1)))
	# PL = 0
	# th_dot = (w - 1)*ws
	# w_dot = 1/(2*H)*(Pm - ps.complex_injections(y[1], y[0]).real[ibus] + PL - Kd*(w-1))
	# Eqp_dot = 1/Td0*(-Eqp - (xd - xdp)*Id + Efd)
	# Edp_dot = 1/Tq0*(-Edp + (xq - xqp)*Iq)
	# dEqp = v*cos(th - d) + Rs*Iq + xdp*Id - Eqp
	# dEdp = v*sin(th - d) + Rs*Id - xqp*Iq - Edp
	# dPg = v*sin(th - d) * Id + v*cos(th - d) * Iq - ps.complex_injections(y[1], y[0]).real[ibus]
	# dQg = v*cos(th - d) * Id - v*sin(th - d) * Iq - ps.complex_injections(y[1], y[0]).imag[ibus]
	dg = ps.dgdx(np.r_[d[1:], v[1:]])
	for iy in range(Ng - 1):
		ix = 8 * iy
		ibus = iy + 1

		for jy in range(2 * (N - 1)):
			#  w_dot
			jac[ix + 1, jy] = (dg[ibus, jy]) / (2 * H[ibus])
			#  dPg
			if jy == ibus + 1:  # if referring to local bus angle
				jac[ix + 6, jy] = -v[ibus] * cos(th[iy] - d[ibus]) * Id[iy] + v[ibus] * sin(th[iy] - d[ibus]) * Iq[iy] - (
					dg[ibus, jy])
			elif jy == N - 1 + ibus:  # if referring to local bus voltage
				jac[ix + 6, jy] = sin(th[iy] - d[ibus]) * Id[iy] + cos(th[iy] - d[ibus]) * Iq[iy] - (dg[ibus, jy])
			else:
				jac[ix + 6, jy] = (dg[ibus, jy])

			#  dQg
			if jy == ibus + 1:  # if referring to local bus angle
				jac[ix + 7, jy] = v[ibus] * sin(th[iy] - d[ibus]) * Id[iy] + v[ibus] * cos(th[iy] - d[ibus]) * Iq[iy] - (
					dg[Ng - 1 + ibus, jy])
			elif jy == N - 1 + ibus:  # if referring to local bus voltage
				jac[ix + 7, jy] = cos(th[iy] - d[ibus]) * Id[iy] - sin(th[iy] - d[ibus]) * Iq[iy] - (dg[Ng - 1 + ibus, jy])
			else:
				jac[ix + 7, jy] = (dg[Ng - 1 + ibus, jy])

		#  dEqp
		jac[ix + 4, ibus - 1] = -v[ibus] * cos(th[iy] - d[ibus])
		jac[ix + 4, N - 1 + ibus - 1] = sin(th[iy] - d[ibus])

		#  dEdp
		jac[ix + 5, ibus - 1] = v[ibus] * sin(th[iy] - d[ibus])
		jac[ix + 5, N - 1 + ibus - 1] = cos(th[iy] - d[ibus])

	return jac


def dyn_C(x, y, df):
	Ng = 4  # TODO: get this number automatically
	ws = df.iloc[0, :]
	H = df.iloc[1, :]
	Kd = df.iloc[2, :]
	Td0 = df.iloc[3, :]
	Tq0 = df.iloc[4, :]
	xd = df.iloc[5, :]
	xdp = df.iloc[6, :]
	xq = df.iloc[7, :]
	xqp = df.iloc[8, :]
	Rs = df.iloc[9, :]

	th = [x[8 * i + 0] for i in range(Ng - 1)]
	w = [x[8 * i + 1] for i in range(Ng - 1)]
	Eqp = [x[8 * i + 2] for i in range(Ng - 1)]
	Edp = [x[8 * i + 3] for i in range(Ng - 1)]
	Iq = [x[8 * i + 4] for i in range(Ng - 1)]
	Id = [x[8 * i + 5] for i in range(Ng - 1)]
	Pm = [x[8 * i + 6] for i in range(Ng - 1)]
	Efd = [x[8 * i + 7] for i in range(Ng - 1)]

	d = y[0]
	v = y[1]

	N = len(d)
	jac = np.zeros((2 * (N - 1), len(x)))
	# PL = 0
	# th_dot = (w - 1)*ws
	# w_dot = 1/(2*H)*(Pm - ps.complex_injections(y[1], y[0]).real[ibus] + PL - Kd*(w-1))
	# Eqp_dot = 1/Td0*(-Eqp - (xd - xdp)*Id + Efd)
	# Edp_dot = 1/Tq0*(-Edp + (xq - xqp)*Iq)
	# dEqp = v*cos(th - d) + Rs*Iq + xdp*Id - Eqp
	# dEdp = v*sin(th - d) + Rs*Id - xqp*Iq - Edp
	# dPg = v*sin(th - d) * Id + v*cos(th - d) * Iq - ps.complex_injections(y[1], y[0]).real[ibus]
	# dQg = v*cos(th - d) * Id - v*sin(th - d) * Iq - ps.complex_injections(y[1], y[0]).imag[ibus]
	for i in range(1, Ng - 1):
		ix = 8 * i
		ibus = i + 1
		# dP/dth
		jac[i, 0] = v[ibus] * cos(th[i] - d[ibus]) * Id[i] - v[ibus] * sin(th[i] - d[ibus]) * Iq[i]
		# dQ/dth
		jac[i + N - 1, 0] = -v[ibus] * sin(th[i] - d[ibus]) * Id[i] - v[ibus] * cos(th[i] - d[ibus]) * Iq[i]

		# dP/dIq
		jac[i, 4] = v[ibus] * cos(th[i] - d[ibus])
		# dQ/dIq
		jac[i + N - 1, 4] = -v[ibus] * sin(th[i] - d[ibus])

		# dP/dId
		jac[i, 5] = v[ibus] * sin(th[i] - d[ibus])
		# dQ/dId
		jac[i + N - 1, 5] = v[ibus] * cos(th[i] - d[ibus])
	return jac


def dyn_D(x, y, df, ps):
	Ng = 4  # TODO: get this number automatically
	ws = df.iloc[0, :]
	H = df.iloc[1, :]
	Kd = df.iloc[2, :]
	Td0 = df.iloc[3, :]
	Tq0 = df.iloc[4, :]
	xd = df.iloc[5, :]
	xdp = df.iloc[6, :]
	xq = df.iloc[7, :]
	xqp = df.iloc[8, :]
	Rs = df.iloc[9, :]

	th = [x[8 * i + 0] for i in range(Ng - 1)]
	w = [x[8 * i + 1] for i in range(Ng - 1)]
	Eqp = [x[8 * i + 2] for i in range(Ng - 1)]
	Edp = [x[8 * i + 3] for i in range(Ng - 1)]
	Iq = [x[8 * i + 4] for i in range(Ng - 1)]
	Id = [x[8 * i + 5] for i in range(Ng - 1)]
	Pm = [x[8 * i + 6] for i in range(Ng - 1)]
	Efd = [x[8 * i + 7] for i in range(Ng - 1)]

	d = y[0]
	v = y[1]

	N = len(d)
	jac = np.zeros((2 * (N - 1), 2 * (N - 1)))

	# dPg = v*sin(th - d) * Id + v*cos(th - d) * Iq - ps.complex_injections(y[1], y[0]).real[ibus]
	# dQg = v*cos(th - d) * Id - v*sin(th - d) * Iq - ps.complex_injections(y[1], y[0]).imag[ibus]
	dg = ps.dgdx(np.r_[d[1:], v[1:]])
	for ibus in range(1, N):
		iy = ibus - 1
		ip = iy
		iq = iy + N - 1
		for jy in range(2 * (N - 1)):
			if ibus in ps.pv:
				#  dPg
				if jy == iy:  # if referring to local bus angle
					jac[iy, jy] = -v[ibus] * cos(th[iy] - d[ibus]) * Id[iy] + v[ibus] * sin(th[iy] - d[ibus]) * Iq[iy] - (
						dg[iy, jy])
				elif jy == iy + N - 1:  # if referring to local bus voltage
					jac[iy, jy] = sin(th[iy] - d[ibus]) * Id[iy] + cos(th[iy] - d[ibus]) * Iq[iy] - (dg[iy, jy])
				else:
					jac[iy, jy] = (dg[iy, jy])

				#  dQg
				if jy == iy:  # if referring to local bus angle  TODO: fix indexing
					jac[iy + N - 1, jy] = v[ibus] * sin(th[iy] - d[ibus]) * Id[iy] + v[ibus] * cos(th[iy] - d[ibus]) * Iq[iy] - (
						dg[iy + N - 1, jy])
				elif jy == iy + N - 1:  # if referring to local bus voltage
					jac[iy + N - 1, jy] = cos(th[iy] - d[ibus]) * Id[iy] - sin(th[iy] - d[ibus]) * Iq[iy] - (dg[iy + N - 1, jy])
				else:
					jac[iy + N - 1, jy] = (dg[iy + N - 1, jy])
			else:
				jac[iy, jy] = (dg[iy, jy])
				jac[iy + N - 1, jy] = (dg[iy + N - 1, jy])
	return jac


if __name__ == "__main__":  # TODO: Vref and Pref should be calculated during initialization
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
	# case_name = "2BUS.txt"
	# case_name = 'KundurEx12-6.txt'
	case_name = 'Kundur_modified.txt'
	dps = DynamicSystem(case_name, udict, sparse=False)
	v0, d0 = dps.flat_start()
	v_nr, d_nr, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False)

	d = {
		'G1': [ws, H1, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs],
		'G2': [ws, H1, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs],
		'G3': [ws, H3, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs],
		'G4': [ws, H3, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs]
	}

	kundur_dyn = pd.DataFrame(d)
	# u = [ws, H, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs]

	xsys = np.array([])
	xss = np.array([])
	y = [d_nr, v_nr]
	print(y)
	Ng = 4
	for bus in range(1, Ng):
		ibus = bus - 1
		th0 = d_nr[bus]
		w0 = 1
		Eqp0 = v_nr[bus]
		Edp0 = 0
		Iq0 = 0.5
		Id0 = 0
		Pm0 = 0.5
		Efd0 = v_nr[bus]
		x0 = [th0, w0, Eqp0, Edp0, Iq0, Id0, Pm0, Efd0]

		x = fsolve(dynamic_eqns, x0, args=(y, kundur_dyn, dps, bus))
		xsys = np.r_[xsys, x]
		xss = np.r_[xss, x[0], x[1], x[2], x[3], x[7], x[6]]
		th = x[0]
		w = x[1]
		Eqp = x[2]
		Edp = x[3]
		Iq = x[4]
		Id = x[5]
		d = y[0][1]
		vq = -(Rs * Iq + xdp * Id - Eqp)
		vd = -(Rs * Id - xqp * Iq - Edp)
		v = (vd ** 2 + vq ** 2) ** (1 / 2)
		print(
			'\nth = ', x[0] * 180 / pi,
			'\nw = ', x[1],
			'\nEqp = ', x[2],
			'\nEdp = ', x[3],
			'\nIq = ', x[4],
			'\nId = ', x[5],
			'\nPm = ', x[6],
			'\nEfd = ', x[7],
			'\nv = ', v

		)
		print('mismatch', dynamic_eqns(x, y, kundur_dyn, dps, bus))

	th0 = d_nr[dps.pv]
	w0 = np.ones(len(dps.pv))
	Eqp0 = v_nr[dps.pv]
	Edp0 = np.zeros(len(dps.pv))
	va0 = v_nr[dps.pv]
	Pm0 = dps.Pc
	x0 = np.array([])
	for i in range(len(dps.pv)):
		x0 = np.r_[x0, th0[i], w0[i], Eqp0[i], Edp0[i], va0[i], Pm0[i]]
	y0 = np.r_[d_nr[1:], v_nr[1:]]
	z0 = np.r_[xss, y0]

	z_eq = fsolve(dps.dyn1, z0)
	print('z_eq', z_eq)
	z_error = dps.dyn1(z_eq)
	print('error', z_error)
	print('z_eq', z_eq)
	xe = z_eq[0:len(x0)]
	ye = z_eq[len(x0):]
	xdot = dps.dyn1_f(xe, ye)
	ydot = dps.dyn1_g(xe, ye)
	J = dps.J_dyn(xe, ye)
	ev, vl, vr = eig(J, left = True)
	print(ev)
	plt.plot(ev.real, ev.imag, '.')
	plt.grid(True, which='both')
	plt.show()

	# print(xsys)
	# A = dyn_A(xsys, y, kundur_dyn)
	# B = dyn_B(xsys, y, kundur_dyn, ps)
	# C = dyn_C(xsys, y, kundur_dyn)
	# D = dyn_D(xsys, y, kundur_dyn, ps)
	# J = A - B @ inv(D) @ C
	# evA, _ = eig(A)
	# evJ, _ = eig(J)
	# dg = ps.dgdx(np.r_[d_nr[1:], v_nr[1:]])
	# print('evA:\n', np.sort(evA.real))
	# print('evJ:\n', np.sort(evJ.real ))
# fig = plt.figure()
# ax = fig.add_subplot(111, polar=True)
# ax.set_thetamin(0)
# ax.set_thetamax(150)
#
# angle = pi/2 - th
# print('ref angle: ', angle*180/pi)
# plt.polar([angle, angle], [0, 1], label="ref")
#
# angle = pi/2 - (th - d_nr[bus - 1])
# print('v angle: ', angle*180/pi)
# plt.polar([angle, angle], [0, v], label="v")
# angle = np.angle(Edp + 1j*Eqp)
# print('E angle: ', angle*180/pi)
# mag = np.abs(Edp + 1j*Eqp)
# plt.polar([angle, angle], [0, mag], label="E")
# # plt.polar([pi/2, pi/2], [0, vq], label="vq")
# # plt.polar([0, 0], [0, vd], label="vd")
# # plt.polar([pi/2, pi/2], [0, Eqp], label="Eqp")
# # plt.polar([0, 0], [0, Edp], label="Edp")
# ax.legend()
#
# plt.show()
