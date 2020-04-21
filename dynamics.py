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
		self.pf_initialized = False
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
		self.Tw = u['Tw']
		self.K1 = u['K1']
		self.T1 = u['T1']
		self.T2 = u['T2']
		self.comp = u['comp']  # 1 or 0 Multiplied by Vs before being added to the AVR. For turning off compensation.
		self.constZ = u['constZ']
		self.constI = u['constI']
		self.constP = u['constP']
		self.Pc = self.bus_data[self.pv, self.busGenMW] / self.p_base
		self.vref = self.bus_data[self.pv, self.busDesiredVolts]
		self.v_desired = self.bus_data[self.pv, self.busDesiredVolts]
		self.vslack = self.bus_data[0, self.busDesiredVolts]
		self.y_gen = self.makeygen()
		self.h = 10**-8

	def pf_init(self, prec=7, maxit=10, qlim=False, verbose=False):
		self.pf_initialized = True
		v0, d0 = self.flat_start()
		v, d, it = self.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
		self.v = v
		self.d = d

	def p_load(self, _v):
		if not self.pf_initialized:
			self.pf_init()
		return self.p_load_full*(self.constP + self.constI / self.v * _v + self.constZ / self.v ** 2 * _v ** 2)

	def q_load(self, _v):
		if not self.pf_initialized:
			self.pf_init()
		return self.q_load_full*(self.constP + self.constI / self.v * _v + self.constZ / self.v ** 2 * _v ** 2)

	def makeygen(self):
		n = len(self.bus_data[:, 0])
		if not self.pf_initialized:
			self.pf_init()
		v = self.v

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
		branch_data_red[nb:, self.branchR] = self.Rs
		branch_data_red[nb:, self.branchX] = self.xp
		y_net = self.makeybus(override=(bus_data_red, branch_data_red))
		ng = len(self.pv) + 1  # include slack
		y11 = y_net[:ng, :ng]
		y12 = y_net[:ng, ng:]
		y21 = y_net[ng:, :ng]
		y22 = y_net[ng:, ng:]
		y_gen = y11 - y12 @ inv(y22) @ y21
		return y_gen

	def type1_init(self):
		if not self.pf_initialized:
			self.pf_init()
		v = self.v
		d = self.d
		s = self.complex_injections(v, d)
		s_l = self.p_load_full + 1j * self.q_load_full
		s_g = s + s_l
		Pm = s_g.real[self.pv]
		Qg = s_g.imag[self.pv]
		v_ = (v * np.exp(1j * d))  # voltage phasor
		I_g = np.conj(s_g / v_)

		th0 = np.array([0.7, 0.4, 0.3])
		w0 = np.ones_like(th0)
		Iq0 = (I_g[self.pv] * 1*np.exp(1j*(pi/2 - th0))).imag
		Id0 = (I_g[self.pv] * 1*np.exp(1j*(pi/2 - th0))).real
		Efd0 = v[self.pv]
		x0 = np.r_[th0, w0, Id0, Iq0, Efd0]
		init_dict = {}
		def mismatch(x):
			n = len(self.pv)

			th = x[:n]
			w = x[n:2*n]
			Id = x[2*n:3*n]
			Iq = x[3*n:4*n]
			Efd = x[4*n:]
			vd = v[self.pv] * sin(th - d[self.pv])
			vq = v[self.pv] * cos(th - d[self.pv])
			Pg = vd * Id + vq * Iq
			Edp = vd - self.xqp * Iq
			Eqp = vq + self.xdp * Id


			th_dot = (w - 1) * self.ws
			w_dot = 1 / (2 * self.H) * (Pm - Pg - self.Kd * (w - 1))  # Pg = vd * Id + vq * Iq
			Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xdp) * Id + Efd)
			Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xqp) * Iq)
			Qg_mis = vq * Id - vd * Iq - Qg
			return np.r_[th_dot, w_dot, Eqp_dot, Edp_dot, Qg_mis]

		x = fsolve(mismatch, x0)
		#
		n = len(self.pv)
		th = x[:n]
		w = x[n:2 * n]
		Id = x[2 * n:3 * n]
		Iq = x[3 * n:4 * n]
		Efd = x[4 * n:]
		vd = v[self.pv] * sin(th - d[self.pv])
		vq = v[self.pv] * cos(th - d[self.pv])
		vref = Efd/self.Ka + v[self.pv]
		Edp = vd - self.xqp * Iq
		Eqp = vq + self.xdp * Id
		Pc = Pm/self.Ksg
		init_dict['th'] = th
		init_dict['Pg'] = Pm
		init_dict['Qg'] = Qg
		init_dict['Eqp'] = Eqp
		init_dict['Edp'] = Edp
		init_dict['Efd'] = Efd
		init_dict['Pc'] = Pc
		init_dict['vref'] = vref
		return init_dict

	def type1_init_bad(self):
		v0, d0 = self.flat_start()
		v, d, it = self.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
		s = self.complex_injections(v, d)
		s_l = self.p_load_full + 1j * self.q_load_full
		s_g = s + s_l
		Pm = s_g.real[self.pv]
		v_ = (v * np.exp(1j * d))  # voltage phasor
		I_g = np.conj(s_g / v_)
		init_dict = {}
		def mismatch(th, init_dict):
			Id = (I_g[self.pv] * 1*np.exp(1j*(pi/2 - th))).real
			Iq = (I_g[self.pv] * 1*np.exp(1j*(pi/2 - th))).imag
			vd = v[self.pv] * sin(th - d[self.pv])
			vq = v[self.pv] * cos(th - d[self.pv])
			Pg = vd * Id + vq * Iq
			Qg = vq * Id - vd * Iq
			Efd = vq + self.xd * Id
			Eqp = vq + self.xdp * Id  # Rs = 0
			Edp = vd - self.xqp * Iq
			Pc = Pm/self.Ksg
			vref = Efd/self.Ka + v[self.pv]
			'''
			th_dot = (w - 1) * self.ws
			w_dot = 1 / (2 * self.H) * (Pm - Pg - self.Kd * (w - 1))  # Pg = vd * Id + vq * Iq
			Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xdp) * Id + Efd)
			Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xqp) * Iq)
			Efd_dot = 1 / self.Ta * (-Efd + self.Ka * (vref - v[self.pv]))
			Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
			'''
			mis = Pm - Pg
			# 0 = -Eqp(th) - (xd - xdp) * Id(th) + Efd(th)
			# 0 = -Edp(th) + (xq - xqp) * Iq(th)
			init_dict['Pg'] = Pg
			init_dict['Qg'] = Qg
			init_dict['Eqp'] = Eqp
			init_dict['Edp'] = Edp
			init_dict['Efd'] = Efd
			init_dict['Pc'] = Pc
			init_dict['vref'] = vref
			return mis
		th0 = d[self.pv]
		th = fsolve(mismatch, th0, args=(init_dict))
		init_dict['th'] = th
		return init_dict

	def type2_init(self):
		if not self.pf_initialized:
			self.pf_init()
		v = self.v
		d = self.d
		s = self.complex_injections(v, d)
		s_l = self.p_load_full + 1j * self.q_load_full
		s_g = s + s_l
		Pm = s_g.real[self.pv]
		Qg = s_g.imag[self.pv]
		v_ = (v * np.exp(1j * d))  # voltage phasor
		I_g = np.conj(s_g / v_)

		th0 = np.array([0.7, 0.4, 0.3])
		w0 = np.ones_like(th0)
		Iq0 = (I_g[self.pv] * 1 * np.exp(1j * (pi / 2 - th0))).imag
		Id0 = (I_g[self.pv] * 1 * np.exp(1j * (pi / 2 - th0))).real
		Efd0 = v[self.pv]
		x0 = np.r_[th0, w0, Id0, Iq0, Efd0]
		init_dict = {}

		def mismatch(x):
			n = len(self.pv)
			th = x[:n]
			w = x[n:2 * n]
			Id = x[2 * n:3 * n]
			Iq = x[3 * n:4 * n]
			Efd = x[4 * n:]
			vd = v[self.pv] * sin(th - d[self.pv])
			vq = v[self.pv] * cos(th - d[self.pv])
			Pg = vd * Id + vq * Iq
			Edp = vd - self.xp * Iq
			Eqp = vq + self.xp * Id

			th_dot = (w - 1) * self.ws
			w_dot = 1 / (2 * self.H) * (Pm - Pg - self.Kd * (w - 1))  # Pg = vd * Id + vq * Iq
			Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xp) * Id + Efd)
			Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xp) * Iq)
			Qg_mis = vq * Id - vd * Iq - Qg
			return np.r_[th_dot, w_dot, Eqp_dot, Edp_dot, Qg_mis]

		x = fsolve(mismatch, x0)
		#
		n = len(self.pv)
		th = x[:n]
		w = x[n:2 * n]
		Id = x[2 * n:3 * n]
		Iq = x[3 * n:4 * n]
		Efd = x[4 * n:]
		vd = v[self.pv] * sin(th - d[self.pv])
		vq = v[self.pv] * cos(th - d[self.pv])
		vref = Efd / self.Ka + v[self.pv]
		Edp = vd - self.xp * Iq
		Eqp = vq + self.xp * Id
		Pc = Pm / self.Ksg
		init_dict['th'] = th
		init_dict['Pg'] = Pm
		init_dict['Qg'] = Qg
		init_dict['Eqp'] = Eqp
		init_dict['Edp'] = Edp
		init_dict['Efd'] = Efd
		init_dict['Pc'] = Pc
		init_dict['vref'] = vref
		return init_dict

	def type3_init(self):
		if not self.pf_initialized:
			self.pf_init()
		v = self.v
		d = self.d
		s = self.complex_injections(v, d)
		s_l = self.p_load_full + 1j * self.q_load_full
		s_g = s + s_l
		Pg = s_g.real[self.pv]
		Qg = s_g.imag[self.pv]
		v_ = (v * np.exp(1j * d))  # voltage phasor
		I_g = np.conj(s_g / v_)
		E = v_[self.pv] + 1j * self.xp * I_g[self.pv]


		vgen = np.r_[v_[self.slack], E]
		Igen = self.y_gen.dot(vgen)
		Pe = (vgen * np.conj(Igen)).real[self.pv]

		th = np.angle(E)
		init_dict = {}
		init_dict['Pg'] = Pg
		init_dict['Qg'] = Qg
		init_dict['E'] = np.abs(E)
		init_dict['th'] = th
		return init_dict

	def dyn1_f_comp(self, x, y, vref, Pc):
		n = len(y) // 2 + 1
		ng = len(self.ws) + 1
		# unpack x
		th = np.array([])
		w = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		va = np.array([])
		Pm = np.array([])
		vw = np.array([])
		vs = np.array([])
		for i in range(ng - 1):
			j = i * 8
			th = np.r_[th, x[0 + j]]
			w = np.r_[w, x[1 + j]]
			Eqp = np.r_[Eqp, x[2 + j]]
			Edp = np.r_[Edp, x[3 + j]]
			va = np.r_[va, x[4 + j]]
			Pm = np.r_[Pm, x[5 + j]]
			vw = np.r_[vw, x[6 + j]]
			vs = np.r_[vs, x[7 + j]]
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
		# Electro-Mechanical States
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pg - self.Kd * (w - 1))  # Pg = vd * Id + vq * Iq
		# Power System Stabilizer
		#  - Washout Filter
		vw_dot = 1/self.Tw*(w_dot*self.Tw - vw)
		#  - Compensator
		vs_dot = 1/self.T2*(vw*self.K1+vw_dot*self.K1*self.T1 - vs)
		# 
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xdp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xqp) * Iq)
		va_dot = 1 / self.Ta * (-Efd + self.Ka * (vref - v[self.pv] + self.comp*vs))
		Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i], vw_dot[i], vs_dot[i]]
		return x_dot


	def dyn1_f(self, x, y, vref, Pc):
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
		va_dot = 1 / self.Ta * (-Efd + self.Ka * (vref - v[self.pv]))
		Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
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

		Pl = self.p_load(v)[self.pvpq]  # excludes slack bus
		Ql = self.q_load(v)[self.pvpq]  # excludes slack bus

		s = (v * np.exp(1j * d)) * np.conj(self.y_bus.dot(v * np.exp(1j * d)))
		# S = P + jQ
		pcalc = s[self.pvpq].real
		qcalc = s[self.pvpq].imag
		dp = Pg - Pl - pcalc
		dq = Qg - Ql - qcalc
		mis = np.concatenate((dp, dq))
		return mis

	def dyn2_f(self, x, vref, Pc):
		# n = len(y) // 2 + 1
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

		vgen = np.r_[self.v[self.slack], (Edp + 1j * Eqp) * (1 * np.exp(1j * (th - pi / 2)))]
		Igen = self.y_gen.dot(vgen)
		v = np.abs(vgen[1:] - Igen[1:] * 1j * self.xp)
		Id = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).real
		Iq = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).imag
		Pe = vgen * np.conj(Igen)
		Pe = Pe.real[self.pv]
		# Pg = vd * Id + vq * Iq
		# Qg = vq * Id - vd * Iq

		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pe - self.Kd * (w - 1))
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xp) * Iq)
		va_dot = 1 / self.Ta * (-Efd + self.Ka * (vref - v))
		Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i]]
		return x_dot

	def dyn2_f_comp(self, x, vref, Pc):
		# n = len(y) // 2 + 1
		ng = len(self.ws) + 1
		# unpack x
		th = np.array([])
		w = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		va = np.array([])
		Pm = np.array([])
		vw = np.array([])
		vs = np.array([])
		for i in range(ng - 1):
			j = i * 8
			th = np.r_[th, x[0 + j]]
			w = np.r_[w, x[1 + j]]
			Eqp = np.r_[Eqp, x[2 + j]]
			Edp = np.r_[Edp, x[3 + j]]
			va = np.r_[va, x[4 + j]]
			Pm = np.r_[Pm, x[5 + j]]
			vw = np.r_[vw, x[6 + j]]
			vs = np.r_[vs, x[7 + j]]
		Efd = va  # no limit equations

		vgen = np.r_[self.v[self.slack], (Edp + 1j * Eqp) * (1 * np.exp(1j * (th - pi / 2)))]
		Igen = self.y_gen.dot(vgen)
		v = np.abs(vgen[1:] - Igen[1:] * 1j * self.xp)
		Id = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).real
		Iq = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).imag
		Pe = vgen * np.conj(Igen)
		Pe = Pe.real[self.pv]
		# Pg = vd * Id + vq * Iq
		# Qg = vq * Id - vd * Iq

		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pe - self.Kd * (w - 1))
		# Power System Stabalizer
		#  - Washout Filter
		vw_dot = 1/self.Tw*(w_dot*self.Tw - vw)
		#  - Compensator
		vs_dot = 1/self.T2*(vw*self.K1+vw_dot*self.K1*self.T1 - vs)
		#
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xp) * Iq)
		va_dot = 1 / self.Ta * (-Efd + self.Ka * (vref - v + vs))
		Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i], vw_dot[i], vs_dot[i]]
		return x_dot

	def dyn3_f(self, x, e_gen, Pm):
		ng = len(self.ws) + 1
		# unpack x
		th = np.array([])
		w = np.array([])
		for i in range(ng - 1):
			j = i * 2
			th = np.r_[th, x[0 + j]]
			w = np.r_[w, x[1 + j]]

		vgen = np.r_[self.v[self.slack], e_gen * np.exp(1j * th)]
		Igen = self.y_gen.dot(vgen)
		Pe = (vgen * np.conj(Igen)).real[self.pv]
		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pe - self.Kd * (w - 1))
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i]]
		return x_dot

	def dyn1(self, z, vref, Pc):
		x = z[0:len(self.pv)*6]
		y = z[len(self.pv)*6:]
		x_dot = self.dyn1_f(x, y, vref, Pc)
		mis = self.dyn1_g(x, y)
		return np.r_[x_dot, mis]

	def A_dyn(self, x_eq, y_eq, vref, Pc):
		A = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				A[i, j] = (self.dyn1_f(x_eq + dx/2, y_eq, vref, Pc)[i] - self.dyn1_f(x_eq - dx/2, y_eq, vref, Pc)[i])/h
				dx[j] = 0
		return A

	def A_dyn_comp(self, x_eq, y_eq, vref, Pc):
		A = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				A[i, j] = (self.dyn1_f_comp(x_eq + dx/2, y_eq, vref, Pc)[i] - self.dyn1_f_comp(x_eq - dx/2, y_eq, vref, Pc)[i])/h
				dx[j] = 0
		return A

	def B_dyn(self, x_eq, y_eq, vref, Pc):
		B = np.zeros((len(x_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				B[i, j] = (self.dyn1_f(x_eq, y_eq + dy/2, vref, Pc)[i] - self.dyn1_f(x_eq, y_eq - dy/2, vref, Pc)[i])/h
				dy[j] = 0
		return B

	def B_dyn_comp(self, x_eq, y_eq, vref, Pc):
		B = np.zeros((len(x_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				B[i, j] = (self.dyn1_f_comp(x_eq, y_eq + dy/2, vref, Pc)[i] - self.dyn1_f_comp(x_eq, y_eq - dy/2, vref, Pc)[i])/h
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

	def J_dyn1(self, x_eq, y_eq, vref, Pc):
		A = self.A_dyn(x_eq, y_eq, vref, Pc)
		B = self.B_dyn(x_eq, y_eq, vref, Pc)
		C = self.C_dyn(x_eq, y_eq)
		D = self.D_dyn(x_eq, y_eq)
		J = A - B @ inv(D) @ C
		return J

	def J_dyn1_comp(self, x_eq, y_eq, vref, Pc):
		A = self.A_dyn_comp(x_eq, y_eq, vref, Pc)
		B = self.B_dyn_comp(x_eq, y_eq, vref, Pc)
		C = self.C_dyn(x_eq, y_eq)
		D = self.D_dyn(x_eq, y_eq)
		J = A - B @ inv(D) @ C
		return J


	def J_dyn2(self, x_eq, vref, Pc):
		J = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				J[i, j] = (self.dyn2_f(x_eq + dx/2, vref, Pc)[i] - self.dyn2_f(x_eq - dx/2, vref, Pc)[i])/h
				dx[j] = 0
		return J
 # f(x)
# df/dx = (f(x + dx/2) - f(x - dx/2))/h
	def J_dyn2_comp(self, x_eq, vref, Pc):
		J = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				J[i, j] = (self.dyn2_f_comp(x_eq + dx/2, vref, Pc)[i] - self.dyn2_f_comp(x_eq - dx/2, vref, Pc)[i])/h
				dx[j] = 0
		return J

	def J_dyn3(self, x_eq, e_gen, Pm):
		J = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				J[i, j] = (self.dyn3_f(x_eq + dx/2, e_gen, Pm)[i] - self.dyn3_f(x_eq - dx/2, e_gen, Pm)[i])/h
				dx[j] = 0
		return J

if __name__ == "__main__":  # TODO: Vref and Pref should be calculated during initialization
	# pbase_gen = 900*10**6
	# vbase_gen = 20*10**3
	# pbase_sys = 100*10**6
	# vbase_sys = 230*10*3
	# zbase_gen = vbase_gen**2/pbase_gen
	# zbase_sys = vbase_sys**2/pbase_sys
	# zbase_conv = zbase_gen/zbase_sys
	zbase_conv = 1/9
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
	v_nr, d_nr, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)

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



