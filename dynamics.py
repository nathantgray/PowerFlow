from scipy.optimize import fsolve
from math import pi
from numpy import sin, cos
import numpy as np
from power_system import PowerSystem
import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib as mplt
from scipy.linalg import eig

class DynamicSystem(PowerSystem):
	def __init__(self, filename, params, sparse=False):
		PowerSystem.__init__(self, filename, sparse=sparse)
		u = params
		self.ws = u['ws']
		self.H = u['H']
		self.Kd = u['Kd']
		self.Td0 = u['Td0']
		self.Tq0 = u['Tq0']
		self.xd = u['xd']
		self.xdp = u['xdp']
		self.xq = u['xq']
		self.xqp = u['xqp']
		self.xp = (self.xqp + self.xdp)/2
		self.Rs = u['Rs']
		self.Ka = u['Ka']
		self.Ta = u['Ta']
		self.Vr_min = u['Vr_min']
		self.Vr_max = u['Vr_max']
		self.Efd_min = u['Efd_min']
		self.Efd_max = u['Efd_max']
		self.Tsg = u['Tsg']
		self.Ksg = u['Ksg']
		self.Psg_min = u['Psg_min']
		self.Psg_max = u['Psg_max']
		self.R = u['R']
		self.Pc = self.bus_data[self.pv, self.busGenMW] / self.p_base
		self.vref = self.bus_data[self.pv, self.busDesiredVolts]
		self.v_desired = self.bus_data[self.pv, self.busDesiredVolts]
		self.vslack = self.bus_data[0, self.busDesiredVolts]
		self.y_gen = self.makeygen()

		self.h = 10**-8

	def makeygen(self):
		n = len(self.bus_data[:, 0])
		v0, d0 = self.flat_start()
		v, d, it = self.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
		# v_ = (v * np.exp(1j * d))  # voltage phasor
		# s = v_ * np.conj(dps.y_bus.dot(v_))
		# modify bus data
		y_load = (self.p_load_full - 1j*self.q_load_full)/v**2
		g_shunt = y_load.real
		b_shunt = y_load.imag
		bus_data = self.bus_data.copy()
		bus_data[:, self.busShuntG] = bus_data[:, self.busShuntG] + g_shunt
		bus_data[:, self.busShuntB] = bus_data[:, self.busShuntB] + b_shunt
		for i in range(n):
			bus_data[i, self.busLoadMW] = 0
			bus_data[i, self.busLoadMVAR] = 0
		bus_data_red = np.vstack((bus_data, bus_data[self.pv, :]))
		for i in range(n, n+len(self.pv)):
			bus_data_red[i, self.busNumber] = i + 1  # set new bus number
			bus_data_red[i, self.busType] = 0  # set bus type to PQ
			bus_data_red[i, self.busGenMW] = 0  # set bus generation to 0
			bus_data_red[i, self.busGenMVAR] = 0  # set bus generation to 0
		nb = len(self.branch_data[:, 0])
		branch_data_red = np.vstack((self.branch_data, np.zeros((len(self.pv), len(self.branch_data[0, :])))))
		for i in range(nb):
			if branch_data_red[i, 0] - 1 in self.pv:
				branch_data_red[i, 0] = branch_data_red[i, 0] + n - 1
			if branch_data_red[i, 1] - 1 in self.pv:
				branch_data_red[i, 1] = branch_data_red[i, 1] + n - 1
		branch_data_red[-3:, 0] = bus_data_red[-3:, 1]
		branch_data_red[-3:, 1] = bus_data_red[-3:, 0]
		for i in range(nb, nb + len(self.pv)):
			branch_data_red[i, 2:5] = [1, 1, 1]
		branch_data_red[nb:, self.branchX] = self.xp
		y_net = self.makeybus(override=(bus_data_red, branch_data_red))
		ng = len(self.pv) + 1  # include slack
		y11 = y_net[:ng, :ng]
		y12 = y_net[:ng, ng:]
		y21 = y_net[ng:, :ng]
		y22 = y_net[ng:, ng:]
		y_gen = y11 - y12 @ inv(y22) @ y21
		return y_gen

	def dyn1_f(self, x, y):
		n = len(y) // 2 + 1
		ng = len(self.ws) + 1
		# unpack x
		th = np.array([])
		w = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		va = np.array([])
		Pm = np.array([])
		for i in range(ng - 1):
			j = i * 6
			th = np.r_[th, x[0 + j]]
			w = np.r_[w, x[1 + j]]
			Eqp = np.r_[Eqp, x[2 + j]]
			Edp = np.r_[Edp, x[3 + j]]
			va = np.r_[va, x[4 + j]]
			Pm = np.r_[Pm, x[5 + j]]
		Efd = va  # no limit equations
		# unpack y
		d = np.r_[0, y[0:n - 1]]
		v = np.r_[self.vslack, y[n - 1:2 * n - 2]]

		# Calculate intermediate values
		vd = v[self.pv] * sin(th - d[self.pv])
		vq = v[self.pv] * cos(th - d[self.pv])
		Id = (Eqp - vq) / self.xdp
		Iq = -(Edp - vd) / self.xqp
		Pg = vd * Id + vq * Iq
		Qg = vq * Id - vd * Iq

		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pg - self.Kd * (w - 1))  # Pg = vd * Id + vq * Iq
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xdp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xqp) * Iq)
		va_dot = 1 / self.Ta * (-Efd + self.Ka * (self.vref - v[self.pv]))
		Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (self.Pc + 1 / self.R * (1 - w)))
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i]]
		return x_dot

	def dyn1_g(self, x, y):
		n = len(y) // 2 + 1
		ng = len(self.ws) + 1
		# unpack x
		th = np.array([])
		w = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		va = np.array([])
		Pm = np.array([])
		for i in range(ng - 1):
			j = i * 6
			th = np.r_[th, x[0 + j]]
			w = np.r_[w, x[1 + j]]
			Eqp = np.r_[Eqp, x[2 + j]]
			Edp = np.r_[Edp, x[3 + j]]
			va = np.r_[va, x[4 + j]]
			Pm = np.r_[Pm, x[5 + j]]
		Efd = va  # no limit equations
		# unpack y
		d = np.r_[0, y[0:n - 1]]
		v = np.r_[self.vslack, y[n - 1:]]

		# Calculate intermediate values
		vd = v[self.pv] * sin(th - d[self.pv])
		vq = v[self.pv] * cos(th - d[self.pv])
		Id = (Eqp - vq) / self.xdp
		Iq = -(Edp - vd) / self.xqp
		Pg_pv = vd * Id + vq * Iq  # only PV buses included
		Qg_pv = vq * Id - vd * Iq  # only PV buses included
		Pg = np.zeros(len(self.pvpq))
		Pg[self.pv - 1] = Pg_pv    # excludes slack bus
		Qg = np.zeros(len(self.pvpq))
		Qg[self.pv - 1] = Qg_pv    # excludes slack bus

		Pl = self.mw_load / self.p_base  # excludes slack bus
		Ql =  self.bus_data[self.pvpq, self.busLoadMVAR] / self.p_base  # excludes slack bus

		s = (v * np.exp(1j * d)) * np.conj(self.y_bus.dot(v * np.exp(1j * d)))
		# S = P + jQ
		pcalc = s[self.pvpq].real
		qcalc = s[self.pvpq].imag
		dp = Pg - Pl - pcalc
		dq = Qg - Ql - qcalc
		mis = np.concatenate((dp, dq))
		return mis

	def dyn1(self, z):
		x = z[0:len(self.pv)*6]
		y = z[len(self.pv)*6:]
		x_dot = self.dyn1_f(x, y)
		mis = self.dyn1_g(x, y)
		return np.r_[x_dot, mis]

	def A_dyn(self, x_eq, y_eq):
		A = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				A[i, j] = (self.dyn1_f(x_eq + dx/2, y_eq)[i] - self.dyn1_f(x_eq - dx/2, y_eq)[i])/h
				dx[j] = 0
		return A

	def B_dyn(self, x_eq, y_eq):
		B = np.zeros((len(x_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				B[i, j] = (self.dyn1_f(x_eq, y_eq + dy/2)[i] - self.dyn1_f(x_eq, y_eq - dy/2)[i])/h
				dy[j] = 0
		return B

	def C_dyn(self, x_eq, y_eq):
		C = np.zeros((len(y_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(y_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				C[i, j] = (self.dyn1_g(x_eq + dx/2, y_eq)[i] - self.dyn1_g(x_eq - dx/2, y_eq)[i])/h
				dx[j] = 0
		return C

	def D_dyn(self, x_eq, y_eq):
		D = np.zeros((len(y_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(y_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				D[i, j] = (self.dyn1_g(x_eq, y_eq + dy/2)[i] - self.dyn1_g(x_eq, y_eq - dy/2)[i])/h
				dy[j] = 0
		return D

	def J_dyn(self, x_eq, y_eq):
		A = self.A_dyn(x_eq, y_eq)
		B = self.B_dyn(x_eq, y_eq)
		C = self.C_dyn(x_eq, y_eq)
		D = self.D_dyn(x_eq, y_eq)
		J = A - B @ inv(D) @ C
		return J

	def dyn3_f(self, x, E_mag, Pm):

		ng = len(self.ws) + 1
		# unpack x
		th = np.array([])
		w = np.array([])
		for i in range(ng - 1):
			j = i * 2
			th = np.r_[th, x[0 + j]]
			w = np.r_[w, x[1 + j]]

		# Calculate intermediate values
		th_full = np.r_[0, th]
		E = E_mag * np.exp(1j * th_full)
		Pg = (E*np.conj(self.y_gen.dot(E))).real
		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pg[self.pv] - self.Kd * (w - 1))

		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i]]
		return x_dot


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
	dps = DynamicSystem(case_name, udict, sparse=True)
	v0, d0 = dps.flat_start()
	v_nr, d_nr, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)

	# xsys = np.array([])
	# xss = np.array([])
	y = [d_nr, v_nr]
	# print(y)
	Ng = 4

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
	z0 = np.r_[x0, y0]

	z_eq = fsolve(dps.dyn1, z0)
	# print('z_eq', z_eq)
	z_error = dps.dyn1(z_eq)
	print('error', max(z_error))
	xe = z_eq[0:len(x0)]
	ye = z_eq[len(x0):]

	# unpack x
	n = len(ye) // 2 + 1
	ng = len(dps.ws) + 1
	th = np.array([])
	w = np.array([])
	Eqp = np.array([])
	Edp = np.array([])
	va = np.array([])
	Pm = np.array([])
	for i in range(ng - 1):
		j = i * 6
		th = np.r_[th, xe[0 + j]]
		w = np.r_[w, xe[1 + j]]
		Eqp = np.r_[Eqp, xe[2 + j]]
		Edp = np.r_[Edp, xe[3 + j]]
		va = np.r_[va, xe[4 + j]]
		Pm = np.r_[Pm, xe[5 + j]]
	# unpack y
	d = np.r_[0, ye[0:n - 1]]
	v = np.r_[dps.vslack, ye[n - 1:]]
	print('th', th)
	print('w', w)
	print('Eqp', Eqp)
	print('Edp', Edp)
	print('va', va)
	print('Pm', Pm)
	print('v', v[:4])
	print('d', d[:4])
	xdot = dps.dyn1_f(xe, ye)
	ydot = dps.dyn1_g(xe, ye)
	J = dps.J_dyn(xe, ye)
	ev, vl, vr = eig(J, left = True)
	print(ev)
	plt.plot(ev.real, ev.imag, '.')
	plt.grid(True, which='both')
	plt.show()



