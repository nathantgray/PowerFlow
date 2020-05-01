from typing import Any, Union

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
	def __init__(self, filename, params, sparse=False, create_ygen=True):
		PowerSystem.__init__(self, filename, sparse=sparse)
		npv = len(self.pv)
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
		self.xp = (self.xqp + self.xdp) / 2
		self.xqp = self.xqp[:npv]
		self.xdp = self.xdp[:npv]
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
		if create_ygen:
			self.y_gen = self.makeygen()
		self.h = 10 ** -8

	def pf_init(self, prec=7, maxit=10, qlim=False, verbose=False):
		self.pf_initialized = True
		v0, d0 = self.flat_start()
		v, d, it = self.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
		self.v = v
		self.d = d

	def p_load(self, _v):
		if not self.pf_initialized:
			self.pf_init()
		return self.p_load_full * (self.constP + self.constI / self.v * _v + self.constZ / self.v ** 2 * _v ** 2)

	def q_load(self, _v):
		if not self.pf_initialized:
			self.pf_init()
		return self.q_load_full * (self.constP + self.constI / self.v * _v + self.constZ / self.v ** 2 * _v ** 2)

	def makeygen(self, v_override=None, gens=None):
		n = len(self.bus_data[:, 0])
		if v_override is None:
			if not self.pf_initialized:
				self.pf_init()
			v = self.v
		else:
			v = v_override
		if gens is None:
			gens = self.pv
		ng = len(gens) + 1
		# modify bus data
		# - Add loads as 100% Z
		y_load = (self.p_load_full - 1j * self.q_load_full) / v ** 2
		g_shunt = y_load.real
		b_shunt = y_load.imag
		bus_data = self.bus_data.copy()
		bus_data[:, self.busShuntG] = bus_data[:, self.busShuntG] + g_shunt
		bus_data[:, self.busShuntB] = bus_data[:, self.busShuntB] + b_shunt

		# - Remove loads from bus data
		for i in range(n):
			bus_data[i, self.busLoadMW] = 0
			bus_data[i, self.busLoadMVAR] = 0
		bus_data_red = np.vstack((bus_data, bus_data[gens, :]))
		# - Add new buses for generators to bus data
		for i in range(n, n + len(gens)):
			bus_data_red[i, self.busNumber] = i + 1  # set new bus number
			bus_data_red[i, self.busType] = 0  # set bus type to PQ
			bus_data_red[i, self.busGenMW] = 0  # set bus generation to 0
			bus_data_red[i, self.busGenMVAR] = 0  # set bus generation to 0
		nb = len(self.branch_data[:, 0])
		branch_data_red = np.vstack((self.branch_data, np.zeros((len(gens), len(self.branch_data[0, :])))))
		for i in range(nb):
			if self.branch_data[i, 0] - 1 in gens:
				k = np.where(gens == self.branch_data[i, 0] - 1)[0]
				branch_data_red[i, 0] = nb + k + 1
			if self.branch_data[i, 1] - 1 in gens:
				k = np.where(gens == self.branch_data[i, 1] - 1)[0]
				branch_data_red[i, 1] = nb + k + 1
		branch_data_red[-ng + 1:, 0] = bus_data_red[-ng + 1:, 1]
		branch_data_red[-ng + 1:, 1] = bus_data_red[-ng + 1:, 0]
		for i in range(nb, nb + len(gens)):
			branch_data_red[i, 2:5] = [1, 1, 1]
		branch_data_red[nb:, self.branchR] = self.Rs
		branch_data_red[nb:, self.branchX] = self.xp
		y_net = self.makeybus(override=(bus_data_red, branch_data_red))
		ng = len(gens) + 1  # include slack
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
		vref = Efd / self.Ka + v[self.pv]
		Edp = vd - self.xqp * Iq
		Eqp = vq + self.xdp * Id
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
			Id = (I_g[self.pv] * 1 * np.exp(1j * (pi / 2 - th))).real
			Iq = (I_g[self.pv] * 1 * np.exp(1j * (pi / 2 - th))).imag
			vd = v[self.pv] * sin(th - d[self.pv])
			vq = v[self.pv] * cos(th - d[self.pv])
			Pg = vd * Id + vq * Iq
			Qg = vq * Id - vd * Iq
			Efd = vq + self.xd * Id
			Eqp = vq + self.xdp * Id  # Rs = 0
			Edp = vd - self.xqp * Iq
			Pc = Pm / self.Ksg
			vref = Efd / self.Ka + v[self.pv]
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
		npv = len(self.pv)
		xp = self.xp[:npv]
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
			Edp = vd - xp * Iq
			Eqp = vq + xp * Id

			th_dot = (w - 1) * self.ws
			w_dot = 1 / (2 * self.H) * (Pm - Pg - self.Kd * (w - 1))  # Pg = vd * Id + vq * Iq
			Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - xp) * Id + Efd)
			Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - xp) * Iq)
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
		Edp = vd - xp * Iq
		Eqp = vq + xp * Id
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
		ng = len(self.pv) + 1
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
		vw_dot = 1 / self.Tw * (w_dot * self.Tw * self.K1 - vw)
		#  - Compensator
		vs_dot = 1 / self.T2 * (vw + vw_dot * self.T1 - vs)
		# Voltage Equations
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - self.xdp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - self.xqp) * Iq)
		# AVR
		va_dot = 1 / self.Ta * (-Efd + self.Ka * (vref - v[self.pv] + vs))
		# Governor
		Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		# Pack and return x_dot array
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[
				x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i], vw_dot[i], vs_dot[i]]
		return x_dot

	def ess(self, x, vs, vs_ref, vdc_ref, vs_dot, P_A_dot, Q_A_dot):
		ng = len(self.pv) + 1
		# unpack x
		k = 1
		r_b = 0.01  # Internal battery resistance
		c_dc = 0.1  # Capacitor on DC sid of converter
		v_b = vdc_ref  # battery voltage
		xs = self.xp[-1]  # temp value
		# upss_m_dot = 0  # temp value
		upss_f_dot = 0  # temp value
		m = x[0]
		phi = x[1]
		v_dc = x[2]
		uw_m = x[3]
		u1_m = x[4]
		upss_m = x[5]
		uw_f = x[6]
		u1_f = x[7]
		upss_f = x[8]
		# gamma = phi + d[7]
		# vs_ = v[7] * np.exp(1j*d[7])
		# vc_ = 1j*xp*Is_ + vs_
		# Calculate intermediate values
		# vs = v[7]
		vc = m*k*v_dc
		# gamma = phi + d[7]
		Isd = vc*sin(phi) / xs
		Isq = (vs - vc*cos(phi)) / xs
		Idc1 = Isd*m*k*cos(phi) + Isq*m*k*sin(phi)
		Idc2 = (v_b - v_dc)/r_b
		# BESS Active Filter
		#  - Washout Filter
		kpss_m = 0.0001898
		Tw_m = 5
		uw_m_dot = 1 / Tw_m * (P_A_dot * Tw_m * kpss_m - uw_m)
		#  - Compensator
		T1_m = 3.309
		T2_m = 0.4
		T3_m = 3.309
		T4_m = 0.4
		u1_m_dot = 1 / T1_m * (uw_m + uw_m_dot * T2_m - u1_m)
		upss_m_dot = 1 / T3_m * (u1_m + u1_m_dot * T4_m - upss_m)
		# BESS Reactive Filter
		#  - Washout Filter
		Tw_f = 5
		kpss_f = 0.05414
		uw_f_dot = 1 / Tw_f * (Q_A_dot * Tw_f * kpss_f - uw_f)
		#  - Compensator
		T1_f = 0.07918
		T2_f = 0.4
		T3_f = 0.07918
		T4_f = 0.4
		u1_f_dot = 1 / T1_f * (uw_f + uw_f_dot * T2_f - u1_f)
		upss_f_dot = 1 / T3_f * (u1_f + u1_f_dot * T4_f - upss_f)

		v_dc_dot = 1/c_dc * (-Idc1 + Idc2)
		m_dot = -0.1*vs_dot + 0.85*(vs_ref - vs) + upss_m_dot
		phi_dot = -0.2*v_dc_dot + 2*(vdc_ref - v_dc) + upss_f_dot

		Ps = v_dc * Idc1

		return np.array([m_dot, phi_dot, v_dc_dot, uw_m_dot, u1_m_dot, upss_m_dot, uw_f_dot, u1_f_dot, upss_f_dot])

	def ess_out(self, x, vs):
		k = 1
		xs = self.xp[-1]
		m = x[0]
		phi = x[1]
		v_dc = x[2]
		vc = m * k * v_dc
		Isd = vc*sin(phi) / xs
		Isq = (vs - vc*cos(phi)) / xs
		Idc1 = Isd*m*k*cos(phi) + Isq*m*k*sin(phi)
		Ps = v_dc * Idc1
		Qs = v_dc * (Isd*m*k*sin(phi) - Isq*m*k*cos(phi))
		return Ps, Qs

	def dyn1ess_f(self, x, y, vref, Pc, vs_ref, vdc_ref, vs_dot, P_A_dot, Q_A_dot, limits=False):
		n = len(y) // 2 + 1
		npv = len(self.pv)
		ng = npv + 1
		xdp = self.xdp[:npv]
		xqp = self.xqp[:npv]
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
		x_ess = x[5 + j + 1:]
		# unpack y
		d = np.r_[0, y[0:n - 1]]
		v = np.r_[self.vslack, y[n - 1:2 * n - 2]]
		# Calculate intermediate values
		vd = v[self.pv] * sin(th - d[self.pv])
		vq = v[self.pv] * cos(th - d[self.pv])
		Id = (Eqp - vq) / xdp
		Iq = -(Edp - vd) / xqp
		Pg = vd * Id + vq * Iq
		Qg = vq * Id - vd * Iq

		# calculate x_dot
		vs = v[7]
		x_ess_dot = self.ess(x_ess, vs, vs_ref, vdc_ref, vs_dot, P_A_dot, Q_A_dot)
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pg - self.Kd * (w - 1))  # Pg = vd * Id + vq * Iq
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - xdp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - xqp) * Iq)
		va_dot_unlim = 1 / self.Ta * (-Efd + self.Ka * (vref - v[self.pv]))
		Psg_dot_unlim = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		va_dot = np.zeros_like(va_dot_unlim)
		Pm_dot = np.zeros_like(Psg_dot_unlim)
		# Limit equations
		if limits:
			for i in range(ng - 1):
				# Exciter non-windup limit
				if va[i] >= self.Vr_max[i] and va_dot_unlim[i] > 0:
					va_dot[i] = 0
				elif va[i] <= self.Vr_min[i] and va_dot_unlim[i] < 0:
					va_dot[i] = 0
				else:
					va_dot[i] = va_dot_unlim[i]

				# Exciter windup limit
				if va[i] > self.Efd_max[i]:
					Efd[i] = self.Efd_max[i]
				elif va[i] < self.Efd_min[i]:
					Efd[i] = self.Efd_min[i]
				else:
					Efd[i] = va[i]

				# Governor windup limit
				if Pm[i] <= self.Psg_min[i] and Psg_dot_unlim[i] < 0:
					Pm_dot[i] = 0
				if Pm[i] >= self.Psg_max[i] and Psg_dot_unlim[i] > 0:
					Pm_dot[i] = 0
				else:
					Pm_dot[i] = Psg_dot_unlim[i]
		else:
			Pm_dot = Psg_dot_unlim
			va_dot = va_dot_unlim
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i]]
		return np.r_[x_dot, x_ess_dot]

	def dyn1ess_g(self, x, y):
		n = len(y) // 2 + 1
		npv = len(self.pv)
		ng = npv + 1
		xdp = self.xdp[:npv]
		xqp = self.xqp[:npv]
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
		x_ess = x[5 + j + 1:]
		# unpack y
		d = np.r_[0, y[0:n - 1]]
		v = np.r_[self.vslack, y[n - 1:]]

		# Calculate intermediate values
		vd = v[self.pv] * sin(th - d[self.pv])
		vq = v[self.pv] * cos(th - d[self.pv])
		Id = (Eqp - vq) / xdp
		Iq = -(Edp - vd) / xqp
		Pg_pv = vd * Id + vq * Iq  # only PV buses included
		Qg_pv = vq * Id - vd * Iq  # only PV buses included
		Ps, Qs = self.ess_out(x_ess, v[7])
		Pg = np.zeros(len(self.pvpq))
		Pg[self.pv - 1] = Pg_pv  # excludes slack bus
		Pg[7] = Ps
		Qg = np.zeros(len(self.pvpq))
		Qg[self.pv - 1] = Qg_pv  # excludes slack bus
		Qg[7] = Qs
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

	def dyn1_f(self, x, y, vref, Pc, limits=False):
		n = len(y) // 2 + 1
		ng = len(self.pv) + 1
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
		va_dot_unlim = 1 / self.Ta * (-Efd + self.Ka * (vref - v[self.pv]))
		Psg_dot_unlim = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		va_dot = np.zeros_like(va_dot_unlim)
		Pm_dot = np.zeros_like(Psg_dot_unlim)
		# Limit equations
		if limits:
			for i in range(ng - 1):
				# Exciter non-windup limit
				if va[i] >= self.Vr_max[i] and va_dot_unlim[i] > 0:
					va_dot[i] = 0
				elif va[i] <= self.Vr_min[i] and va_dot_unlim[i] < 0:
					va_dot[i] = 0
				else:
					va_dot[i] = va_dot_unlim[i]

				# Exciter windup limit
				if va[i] > self.Efd_max[i]:
					Efd[i] = self.Efd_max[i]
				elif va[i] < self.Efd_min[i]:
					Efd[i] = self.Efd_min[i]
				else:
					Efd[i] = va[i]

				# Governor windup limit
				if Pm[i] <= self.Psg_min[i] and Psg_dot_unlim[i] < 0:
					Pm_dot[i] = 0
				if Pm[i] >= self.Psg_max[i] and Psg_dot_unlim[i] > 0:
					Pm_dot[i] = 0
				else:
					Pm_dot[i] = Psg_dot_unlim[i]
		else:
			Pm_dot = Psg_dot_unlim
			va_dot = va_dot_unlim
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i]]
		return x_dot

	def dyn1_g(self, x, y):
		n = len(y) // 2 + 1
		ng = len(self.pv) + 1
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
		Pg[self.pv - 1] = Pg_pv  # excludes slack bus
		Qg = np.zeros(len(self.pvpq))
		Qg[self.pv - 1] = Qg_pv  # excludes slack bus

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

	def dyn1_g_comp(self, x, y):
		n = len(y) // 2 + 1
		ng = len(self.pv) + 1
		# unpack x
		th = np.array([])
		w = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		va = np.array([])
		Pm = np.array([])
		for i in range(ng - 1):
			j = i * 8
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
		Pg[self.pv - 1] = Pg_pv  # excludes slack bus
		Qg = np.zeros(len(self.pvpq))
		Qg[self.pv - 1] = Qg_pv  # excludes slack bus

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

	def dyn2_f_comp(self, x, vref, Pc):
		npv = len(self.pv)
		ng = npv + 1
		xp = self.xp[:npv]
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
		v = np.abs(vgen[1:] - Igen[1:] * 1j * xp)
		Id = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).real
		Iq = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).imag
		Se = vgen * np.conj(Igen)
		Pe = Se.real[self.pv]
		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pe - self.Kd * (w - 1))
		# Power System Stabalizer
		#  - Washout Filter
		vw_dot = 1 / self.Tw * (w_dot * self.Tw - vw)
		#  - Compensator
		vs_dot = 1 / self.T2 * (vw * self.K1 + vw_dot * self.K1 * self.T1 - vs)
		#
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - xp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - xp) * Iq)
		va_dot = 1 / self.Ta * (-Efd + self.Ka * (vref - v + vs))
		Pm_dot = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[
				x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i], vw_dot[i], vs_dot[i]]
		return x_dot

	def dyn2_f(self, x, vref, Pc, limits=False):
		# n = len(y) // 2 + 1
		npv = len(self.pv)
		ng = npv + 1
		xp = self.xp[:npv]
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
		v = np.abs(vgen[1:] - Igen[1:] * 1j * xp)
		Id = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).real
		Iq = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).imag
		Pe = vgen * np.conj(Igen)
		Pe = Pe.real[self.pv]

		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pe - self.Kd * (w - 1))
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - xp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - xp) * Iq)
		va_dot_unlim = 1 / self.Ta * (-Efd + self.Ka * (vref - v))
		Psg_dot_unlim = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		va_dot = np.zeros_like(va_dot_unlim)
		Pm_dot = np.zeros_like(Psg_dot_unlim)
		# Limit equations
		if limits:
			for i in range(ng - 1):
				# Exciter non-windup limit
				if va[i] >= self.Vr_max[i] and va_dot_unlim[i] > 0:
					va_dot[i] = 0
				elif va[i] <= self.Vr_min[i] and va_dot_unlim[i] < 0:
					va_dot[i] = 0
				else:
					va_dot[i] = va_dot_unlim[i]

				# Exciter windup limit
				if va[i] > self.Efd_max[i]:
					Efd[i] = self.Efd_max[i]
				elif va[i] < self.Efd_min[i]:
					Efd[i] = self.Efd_min[i]
				else:
					Efd[i] = va[i]

				# Governor non-windup limit
				if Pm[i] <= self.Psg_min[i] and Psg_dot_unlim[i] < 0:
					Pm_dot[i] = 0
				if Pm[i] >= self.Psg_max[i] and Psg_dot_unlim[i] > 0:
					Pm_dot[i] = 0
				else:
					Pm_dot[i] = Psg_dot_unlim[i]
		else:
			Pm_dot = Psg_dot_unlim
			va_dot = va_dot_unlim
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i]]
		return x_dot

	def dyn2ess_f(self, x, vref, Pc, limits=False):
		# n = len(y) // 2 + 1
		npv = len(self.pv)
		ng = npv + 1
		xp = self.xp[:npv]
		# unpack x
		th = np.array([])
		w = np.array([])
		Eqp = np.array([])
		Edp = np.array([])
		va = np.array([])
		Pm = np.array([])
		for i in range(npv):
			j = i * 6
			th = np.r_[th, x[0 + j]]
			w = np.r_[w, x[1 + j]]
			Eqp = np.r_[Eqp, x[2 + j]]
			Edp = np.r_[Edp, x[3 + j]]
			va = np.r_[va, x[4 + j]]
			Pm = np.r_[Pm, x[5 + j]]
		x_ess = x[5 + j + 1:]
		Efd = va  # no limit equations

		vgen = np.r_[self.v[self.slack], (Edp + 1j * Eqp) * (1 * np.exp(1j * (th - pi / 2)))]
		Igen = self.y_gen.dot(vgen)
		v = np.abs(vgen[1:] - Igen[1:] * 1j * xp)
		Id = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).real
		Iq = (Igen[self.pv] * np.exp(1j * (pi / 2 - th))).imag
		Pe = vgen * np.conj(Igen)
		Pe = Pe.real[self.pv]

		# calculate x_dot
		th_dot = (w - 1) * self.ws
		w_dot = 1 / (2 * self.H) * (Pm - Pe - self.Kd * (w - 1))
		Eqp_dot = 1 / self.Td0 * (-Eqp - (self.xd - xp) * Id + Efd)
		Edp_dot = 1 / self.Tq0 * (-Edp + (self.xq - xp) * Iq)
		va_dot_unlim = 1 / self.Ta * (-Efd + self.Ka * (vref - v))
		Psg_dot_unlim = 1 / self.Tsg * (-Pm + self.Ksg * (Pc + 1 / self.R * (1 - w)))
		va_dot = np.zeros_like(va_dot_unlim)
		Pm_dot = np.zeros_like(Psg_dot_unlim)
		# Limit equations
		if limits:
			for i in range(ng - 1):
				# Exciter non-windup limit
				if va[i] >= self.Vr_max[i] and va_dot_unlim[i] > 0:
					va_dot[i] = 0
				elif va[i] <= self.Vr_min[i] and va_dot_unlim[i] < 0:
					va_dot[i] = 0
				else:
					va_dot[i] = va_dot_unlim[i]

				# Exciter windup limit
				if va[i] > self.Efd_max[i]:
					Efd[i] = self.Efd_max[i]
				elif va[i] < self.Efd_min[i]:
					Efd[i] = self.Efd_min[i]
				else:
					Efd[i] = va[i]

				# Governor non-windup limit
				if Pm[i] <= self.Psg_min[i] and Psg_dot_unlim[i] < 0:
					Pm_dot[i] = 0
				if Pm[i] >= self.Psg_max[i] and Psg_dot_unlim[i] > 0:
					Pm_dot[i] = 0
				else:
					Pm_dot[i] = Psg_dot_unlim[i]
		else:
			Pm_dot = Psg_dot_unlim
			va_dot = va_dot_unlim
		x_dot = np.array([])
		for i in range(ng - 1):
			x_dot = np.r_[x_dot, th_dot[i], w_dot[i], Eqp_dot[i], Edp_dot[i], va_dot[i], Pm_dot[i]]
		return x_dot

	def dyn3_f(self, x, e_gen, Pm):
		ng = len(self.pv) + 1
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
		x = z[0:len(self.pv) * 6]
		y = z[len(self.pv) * 6:]
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
				A[i, j] = (self.dyn1_f(x_eq + dx / 2, y_eq, vref, Pc)[i] - self.dyn1_f(x_eq - dx / 2, y_eq, vref, Pc)[
					i]) / h
				dx[j] = 0
		return A

	def A_dyn_comp(self, x_eq, y_eq, vref, Pc):
		A = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				A[i, j] = (self.dyn1_f_comp(x_eq + dx / 2, y_eq, vref, Pc)[i] -
						   self.dyn1_f_comp(x_eq - dx / 2, y_eq, vref, Pc)[i]) / h
				dx[j] = 0
		return A

	def B_dyn(self, x_eq, y_eq, vref, Pc):
		B = np.zeros((len(x_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				B[i, j] = (self.dyn1_f(x_eq, y_eq + dy / 2, vref, Pc)[i] - self.dyn1_f(x_eq, y_eq - dy / 2, vref, Pc)[
					i]) / h
				dy[j] = 0
		return B

	def B_dyn_comp(self, x_eq, y_eq, vref, Pc):
		B = np.zeros((len(x_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				B[i, j] = (self.dyn1_f_comp(x_eq, y_eq + dy / 2, vref, Pc)[i] -
						   self.dyn1_f_comp(x_eq, y_eq - dy / 2, vref, Pc)[i]) / h
				dy[j] = 0
		return B

	def C_dyn(self, x_eq, y_eq):
		C = np.zeros((len(y_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(y_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				C[i, j] = (self.dyn1_g(x_eq + dx / 2, y_eq)[i] - self.dyn1_g(x_eq - dx / 2, y_eq)[i]) / h
				dx[j] = 0
		return C

	def C_dyn_comp(self, x_eq, y_eq):
		C = np.zeros((len(y_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(y_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				C[i, j] = (self.dyn1_g_comp(x_eq + dx / 2, y_eq)[i] - self.dyn1_g_comp(x_eq - dx / 2, y_eq)[i]) / h
				dx[j] = 0
		return C

	def D_dyn(self, x_eq, y_eq):
		D = np.zeros((len(y_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(y_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				D[i, j] = (self.dyn1_g(x_eq, y_eq + dy / 2)[i] - self.dyn1_g(x_eq, y_eq - dy / 2)[i]) / h
				dy[j] = 0
		return D

	def D_dyn_comp(self, x_eq, y_eq):
		D = np.zeros((len(y_eq), len(y_eq)))
		dy = np.zeros(len(y_eq))
		h = self.h
		for i in range(len(y_eq)):
			for j in range(len(y_eq)):
				dy[j] = h
				D[i, j] = (self.dyn1_g_comp(x_eq, y_eq + dy / 2)[i] - self.dyn1_g_comp(x_eq, y_eq - dy / 2)[i]) / h
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
		C = self.C_dyn_comp(x_eq, y_eq)
		D = self.D_dyn_comp(x_eq, y_eq)
		J = A - B @ inv(D) @ C
		return J

	def J_dyn2(self, x_eq, vref, Pc):
		J = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				J[i, j] = (self.dyn2_f(x_eq + dx / 2, vref, Pc)[i] - self.dyn2_f(x_eq - dx / 2, vref, Pc)[i]) / h
				dx[j] = 0
		return J

	def J_dyn2_comp(self, x_eq, vref, Pc):
		J = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				J[i, j] = (self.dyn2_f_comp(x_eq + dx / 2, vref, Pc)[i] - self.dyn2_f_comp(x_eq - dx / 2, vref, Pc)[
					i]) / h
				dx[j] = 0
		return J

	def J_dyn3(self, x_eq, e_gen, Pm):
		J = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = self.h
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				J[i, j] = (self.dyn3_f(x_eq + dx / 2, e_gen, Pm)[i] - self.dyn3_f(x_eq - dx / 2, e_gen, Pm)[i]) / h
				dx[j] = 0
		return J
