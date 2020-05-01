import numpy as np
import pandas as pd
from copy import deepcopy
from crout_reorder import mat_solve
from sparse import Sparse
from numpy import sin, cos, angle, imag, real

class PowerSystem:
	def __init__(self, filename, sparse=False):
		self.sparse = sparse
		self.busNumber = 0
		self.busArea = 2
		self.busZone = 3
		self.busType = 4
		self.busFinalVoltage = 5
		self.busFinalAngle = 6
		self.busLoadMW = 7
		self.busLoadMVAR = 8
		self.busGenMW = 9
		self.busGenMVAR = 10
		self.busBaseKV = 11
		self.busDesiredVolts = 12
		self.busMaxMVAR = 13
		self.busMinMVAR = 14
		self.busShuntG = 15
		self.busShuntB = 16
		self.busRemoteControlledBusNumber = 17

		self.branchFromBus = 0
		self.branchToBus = 1
		self.branchR = 6
		self.branchX = 7
		self.branchB = 8
		self.branchTurnsRatio = 14
		self.branchPhaseShift = 15

		self.bus_data, self.branch_data, self.p_base = self.read_case(filename)

		# Make the Y-bus matrix
		self.y_bus = self.makeybus()

		# Get bus types
		types = self.bus_data[:, self.busType]
		self.slack = np.where(types == 3)[0]
		self.pv = np.where(types == 2)[0]  # list of PV bus indices
		self.pq = np.where(types < 2)[0]  # list of PQ bus indices
		self.pvpq = np.sort(np.concatenate((self.pv, self.pq)))  # list of indices of non-slack buses
		self.gen_buses = np.where(types == 2)[0]  # list of generator bus indices

		# Calculate scheduled P and Q for each bus
		self.mw_gen = self.bus_data[self.pvpq, self.busGenMW]
		self.mw_load = self.bus_data[self.pvpq, self.busLoadMW]
		self.mvar_load = self.bus_data[self.pq, self.busLoadMVAR]
		self.mw_gen_full = self.bus_data[:, self.busGenMW]
		self.mw_load_full = self.bus_data[:, self.busLoadMW]
		self.p_gen_full = self.mw_gen_full / self.p_base
		self.mvar_load_full = self.bus_data[:, self.busLoadMVAR]
		self.p_load_full = self.mw_load_full / self.p_base
		self.q_load_full = self.mvar_load_full / self.p_base
		self.psched = np.array(self.mw_gen - self.mw_load) / self.p_base
		self.qsched = np.array(- self.mvar_load) / self.p_base
		self.qsched_full = np.array(- self.mvar_load_full) / self.p_base
		self.psched_full = np.array(self.mw_gen_full - self.mw_load_full) / self.p_base
		self.q_lim = np.c_[
			self.bus_data[:, self.busMaxMVAR] / self.p_base,
			self.bus_data[:, self.busMinMVAR] / self.p_base]
		self.q_min_bus = np.array([]).astype(int)
		self.q_max_bus = np.array([]).astype(int)

	@staticmethod
	def read_case(file_name):
		mva_base = 1
		with open(file_name) as f:
			for line in f:
				mva_base = float(line[31:37])
				break

		# count rows of bus data
		i = 0
		bus_rows = 0
		bus_col = 18
		with open(file_name) as f:
			for line in f:
				# Bus data
				if i >= 2:
					if line[0] == '-':
						bus_rows = i - 2
						break
				i = i + 1
		# Build bus data array
		bus_data = np.zeros((bus_rows, bus_col))
		i = 0
		j = 0
		with open(file_name) as f:
			for line in f:
				if i >= 2 and j < bus_rows:
					if line[0] == '-':
						break
					bus_data[j, 0] = int(line[0:4])
					bus_data[j, 1] = int(line[0:4])
					bus_data[j, 2] = int(line[18:20])
					bus_data[j, 3] = int(line[20:23])
					bus_data[j, 4] = int(line[24:26])
					bus_data[j, 5] = float(line[27:33])
					bus_data[j, 6] = float(line[33:40])
					bus_data[j, 7] = float(line[40:49])
					bus_data[j, 8] = float(line[49:59])
					bus_data[j, 9] = float(line[59:67])
					bus_data[j, 10] = float(line[67:75])
					bus_data[j, 11] = float(line[76:83])
					bus_data[j, 12] = float(line[84:90])
					bus_data[j, 13] = float(line[90:98])
					bus_data[j, 14] = float(line[98:106])
					bus_data[j, 15] = float(line[106:114])
					bus_data[j, 16] = float(line[114:122])
					bus_data[j, 17] = int(line[123:127])

					j = j + 1
				i = i + 1

		branchDataStart = bus_rows + 4
		i = 0
		branch_rows = 0
		branch_col = 21
		with open(file_name) as f:
			for line in f:
				# Bus data
				if i >= branchDataStart:
					if line[0] == '-':
						branch_rows = i - branchDataStart
						break
				i = i + 1
		branch_data = np.zeros((branch_rows, branch_col))
		i = 0
		j = 0
		with open(file_name) as f:
			for line in f:
				if i >= branchDataStart and j < branch_rows:
					if line[0] == '-':
						break
					branch_data[j, 0] = int(line[0:4])  # Columns  1- 4   Tap bus number (I) *
					branch_data[j, 1] = int(line[5:9])  # Columns  6- 9   Z bus number (I) *
					branch_data[j, 2] = int(line[10:12])  # Columns 11-12   Load flow area (I)
					branch_data[j, 3] = int(line[12:15])  # Columns 13-14   Loss zone (I)
					branch_data[j, 4] = int(line[16:17])  # Column  17      Circuit (I) * (Use 1 for single lines)
					branch_data[j, 5] = int(line[18:19])  # Column  19      Type (I) *
					branch_data[j, 6] = float(line[19:29])  # Columns 20-29   Branch resistance R, per unit (F) *
					branch_data[j, 7] = float(line[29:40])  # Columns 30-40   Branch reactance X, per unit (F) *
					branch_data[j, 8] = float(line[40:50])  # Columns 41-50   Line charging B, per unit (F) *
					branch_data[j, 9] = int(line[50:55])  # Columns 51-55   Line MVA rating No 1 (I) Left justify!
					branch_data[j, 10] = int(line[56:61])  # Columns 57-61   Line MVA rating No 2 (I) Left justify!
					branch_data[j, 11] = int(line[62:67])  # Columns 63-67   Line MVA rating No 3 (I) Left justify!
					branch_data[j, 12] = int(line[68:72])  # Columns 69-72   Control bus number
					branch_data[j, 13] = int(line[73:74])  # Column  74      Side (I)
					branch_data[j, 14] = float(line[75:82])  # Columns 77-82   Transformer final turns ratio (F)
					branch_data[j, 15] = float(
						line[83:90])  # Columns 84-90   Transformer (phase shifter) final angle (F)
					branch_data[j, 16] = float(line[90:97])  # Columns 91-97   Minimum tap or phase shift (F)
					branch_data[j, 17] = float(line[97:104])  # Columns 98-104  Maximum tap or phase shift (F)
					branch_data[j, 18] = float(line[105:111])  # Columns 106-111 Step size (F)
					branch_data[j, 19] = float(line[112:118])  # Columns 113-119 Minimum voltage, MVAR or MW limit (F)
					branch_data[j, 20] = float(line[119:126])  # Columns 120-126 Maximum voltage, MVAR or MW limit (F)

					j = j + 1
				i = i + 1
		return bus_data, branch_data, mva_base

	def makeybus(self, make_bpp=False, make_bp=False, override=None):
		# Produces the Y bus matrix of a power system.
		# Written by Nathan Gray
		if isinstance(override, tuple):
			bus_data = override[0]
			branch_data = override[1]
		else:
			bus_data = self.bus_data
			branch_data = self.branch_data
		busShuntG = 15
		busShuntB = 16

		branchR = 6
		branchX = 7
		branchB = 8
		branchTurnsRatio = 14
		branchPhaseShift = 15

		nl = branch_data.shape[0]  # number of lines
		n = bus_data.shape[0]  # number of buses
		# Prepare data for algorithm
		if make_bp:
			z = branch_data[:, branchX] * 1j
		else:
			z = branch_data[:, branchR] + branch_data[:, branchX] * 1j
		y = z ** -1
		b_line = branch_data[:, branchB]
		if make_bp:
			ratio = np.ones(nl)
		else:
			ratio = np.where(branch_data[:, branchTurnsRatio] == 0.0, 1, branch_data[:, branchTurnsRatio])
		if make_bpp:
			shift = np.zeros(nl)
		else:
			shift = np.radians(branch_data[:, branchPhaseShift])
		t = ratio * np.cos(shift) + 1j * ratio * np.sin(shift)
		# Shunt admittances for each bus.
		y_shunt = bus_data[:, busShuntG] + 1j * bus_data[:, busShuntB]
		frombus = branch_data[:, 0]
		tobus = branch_data[:, 1]

		if self.sparse:
			y_bus = Sparse.zeros((n, n), dtype=complex)  # initialize Y Bus Matrix
		else:
			y_bus = np.zeros((n, n), dtype=complex)  # initialize Y Bus Matrix
		# The following algorithm takes the arguments: y, b_line, t, y_shunt
		# Create the four entries of a Y-Bus matrix for each line.
		#
		# i|-}|{--~~--|j
		# 	 t:1   y
		#
		# [y/|t|^2   -y/t*]
		# [-y/t      y  ]

		yjj = y + 1j * b_line / 2
		yii = yjj / (abs(t) ** 2)
		yij = -y / np.conj(t)
		yji = -y / t

		for k in range(nl):
			i = int(frombus[k]) - 1
			j = int(tobus[k]) - 1
			y_bus[i, j] = yij[k]
			y_bus[j, i] = yji[k]
			y_bus[i, i] += yii[k]
			y_bus[j, j] += yjj[k]
		if not make_bp:
			for i in range(n):
				y_bus[i, i] += y_shunt[i]

		return y_bus

	# ~~~~~ Power Flows ~~~~~
	def flat_start(self):
		# Initialize with flat start
		v_flat = np.array(
			np.where(self.bus_data[:, self.busDesiredVolts] == 0.0, 1, self.bus_data[:, self.busDesiredVolts]))
		d_flat = np.zeros(v_flat.shape)
		return v_flat, d_flat

	@staticmethod
	def pf_dc(d_start, y, pvpq, psched, lam=None):
		d = deepcopy(d_start)
		bdc = -y.imag[pvpq, :][:, pvpq]
		if lam is not None:
			d[pvpq] = mat_solve(bdc, lam * psched)
		else:
			d[pvpq] = mat_solve(bdc, psched)
		return d

	def pf_newtonraphson(self, v_start, d_start, prec=2, maxit=4, qlim=True, qlim_prec=2, lam=None, verbose=True, debug_file=None):
		# Uses Newton-Raphson method to solve the power-flow of a power system.
		# Written by Nathan Gray
		# Arguments:
		# v_start: list of voltage magnitudes in system
		# d_start: list of voltage phase angles in system
		# prec: program finishes when all mismatches < 10^-abs(prec)
		# maxit: maximum number of iterations
		if verbose:
			print("\n~~~~~~~~~~ Start Newton-Raphson Method ~~~~~~~~~~\n")
		psched = deepcopy(self.psched)
		qsched = deepcopy(self.qsched)
		if lam is not None:
			psched = lam * psched
			qsched = lam * qsched
		v = deepcopy(v_start)
		d = deepcopy(d_start)
		y = self.y_bus
		pvpq = self.pvpq
		pq = deepcopy(self.pq)
		pv = deepcopy(self.pv)
		pq_last = deepcopy(pq)
		n = np.shape(y)[0]

		if debug_file is not None:
			results = []
			df_space = pd.DataFrame(data={"": [""]})

		i = 0
		# Newton Raphson
		for i in range(maxit + 1):
			# Calculate Mismatches
			mis, p_calc, q_calc = self.mismatch(v, d, y, pq, pvpq, psched, qsched)
			if debug_file is not None:
				results.append(self.results2df(v, d))
				results.append(pd.DataFrame(data={"It: {}, E: {:.2E}".format(i, max(abs(mis))): [""]}))
				results.append(df_space)
			if verbose:
				print("error: ", max(abs(mis)))
			pq_last = deepcopy(pq)
			if qlim and max(abs(mis)) < 10 ** -abs(qlim_prec):
				# Check Limits
				pv, pq, qsched = self.check_limits(v, d, y, pv, pq)
				# Calculate Mismatches
				mis, p_calc, q_calc = self.mismatch(v, d, y, pq, pvpq, psched, qsched)
			# Check error
			if max(abs(mis)) < 10 ** -abs(prec) and np.array_equiv(pq_last, pq):
				if verbose:
					print("Newton Raphson completed in ", i, " iterations.")
				# pv, pq, qsched = self.check_limits(v, d, y, pv, pq)
				if debug_file is not None:
					pd.concat(results, axis=1, sort=False).to_csv(debug_file, float_format='%.3f')
				return v, d, i
			# Calculate Jacobian
			j = self.pf_jacobian(v, d, pq)
			# Calculate update values
			dx = mat_solve(j, mis)
			# Update angles: d_(n+1) = d_n + dd
			d[pvpq] = d[pvpq] + dx[:n - 1]
			# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
			v[pq] = v[pq] * (1 + dx[n - 1:n + pq.size - 1])


		if debug_file is not None:
			pd.concat(results, axis=1, sort=False).to_csv(debug_file, float_format='%.3f')
		# print(v, d)
		if verbose:
			print("Max iterations reached, ", i, ".")
		return v, d, i

	def pf_fast_decoupled(self, v_start, d_start, prec=2, maxit=100, qlim=True, qlim_prec=2, debug_file=None):
		# Uses Fast Decoupled method to solve the power-flow of a power system.
		# Written by Nathan Gray
		# Arguments:
		# v_start: list of voltage magnitudes in system
		# d_start: list of voltage phase angles in system
		# prec: program finishes when all mismatches < 10^-abs(prec)
		# maxit: maximum number of iterations
		print("\n~~~~~~~~~~ Start Fast Decoupled Method ~~~~~~~~~~\n")
		psched = self.psched
		qsched = deepcopy(self.qsched)
		y = self.y_bus
		bp = self.makeybus(make_bp=True)
		bpp = self.makeybus(make_bpp=True)
		v = deepcopy(v_start)
		d = deepcopy(d_start)
		pvpq = self.pvpq
		pq = self.pq
		pv = self.pv
		pq_last = deepcopy(pq)
		# Decoupled Power Flow
		bd = -bp.imag[pvpq, :][:, pvpq]
		bv = -bpp.imag[pq, :][:, pq]    # TODO: Trouble here for certain combinations of PV buses on Kundur system
		# bd = self.pf_jacobian(v, d, pq, decoupled=True)[0]
		# bv = self.pf_jacobian(v, d, pq, decoupled=True)[1]

		if debug_file is not None:
			results = []
			df_space = pd.DataFrame(data={"": [""]})
			pd.DataFrame(data=y.real).to_csv('yreal_'+debug_file, float_format='%.3f')
			pd.DataFrame(data=y.imag).to_csv('yimag_'+debug_file, float_format='%.3f')
			pd.DataFrame(data=bd).to_csv('bd_'+debug_file, float_format='%.3f')
			pd.DataFrame(data=bv).to_csv('bv_'+debug_file, float_format='%.3f')

		i = 0
		for i in range(maxit + 1):
			# Calculate Mismatches
			mis, p_calc, q_calc = self.mismatch(v, d, y, pq, pvpq, psched, qsched)
			if debug_file is not None:
				results.append(self.results2df(v, d))
				results.append(pd.DataFrame(data={"It: {}, E: {:.2E}".format(i, max(abs(mis))): [""]}))
				results.append(df_space)
			print("error: ", max(abs(mis)))
			pq_last = deepcopy(pq)
			if qlim and max(abs(mis)) < 10 ** -abs(qlim_prec):  # Do q-limit check
				pv, pq, qsched = self.check_limits(v, d, y, pv, pq)
				# Calculate Mismatches
				mis, p_calc, q_calc = self.mismatch(v, d, y, pq, pvpq, psched, qsched)
			# Only update bv matrix size if pq changes
			if not np.array_equiv(pq_last, pq):
				bv = -bpp.imag[pq, :][:, pq]
			# Check error
			if max(abs(mis)) < 10 ** -abs(prec) and np.array_equiv(pq_last, pq):
				print("Decoupled Power Flow completed in ", i, " iterations.")
				if debug_file is not None:
					pd.concat(results, axis=1, sort=False).to_csv(debug_file, float_format='%.3f')
				return v, d, i
			d[pvpq] = d[pvpq] + mat_solve(bd, mis[0:len(pvpq)] / v[pvpq])
			# mis, p_calc, q_calc = self.mismatch(v, d, y, pq, pvpq, psched, qsched)
			v[pq] = v[pq] + mat_solve(bv, mis[len(pvpq):] / v[pq])


		if debug_file is not None:
			pd.concat(results, axis=1, sort=False).to_csv(debug_file, float_format='%.3f')

		print("Max iterations reached, ", i, ".")
		return v, d, i

	def results2df(self, v, d):
		s = (v * np.exp(1j * d)) * np.conj(self.y_bus.dot(v * np.exp(1j * d)))
		sol_dic = {
			'Bus': self.bus_data[:, 0],
			"Type": self.bus_data[:, 4],
			"V Result": v,
			"Angle Result": d * 180 / np.pi + self.bus_data[0, 6],
			"MW Injected": s.real * self.p_base,
			"MVAR Injected": s.imag * self.p_base}
		df = pd.DataFrame(data=sol_dic)
		return df

	def check_limits(self, v, d, y, pv, pq):
		q_lim = self.q_lim
		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)

		# Find buses that are no longer limited and convert them back to PV buses
		if len(self.q_max_bus) > 0 or len(self.q_min_bus) > 0:
			if len(self.q_max_bus) > 0:
				for bus in self.q_max_bus:
					if v[bus] > self.bus_data[bus, self.busDesiredVolts]:
						# Bus is no longer limited, make PV bus again.
						pq = np.setdiff1d(pq, [bus])
						pv = np.unique(np.concatenate((pv, [bus])))
						self.q_max_bus = np.delete(self.q_max_bus, np.where(self.q_max_bus == bus))
						print("Not Q Limited: ", bus, " because ", v[bus], " > ",
							  self.bus_data[bus, self.busDesiredVolts])
			if len(self.q_min_bus) > 0:
				for bus in self.q_min_bus:
					if v[bus] < self.bus_data[bus, self.busDesiredVolts]:
						# Bus is no longer limited, make PV bus again.
						pq = np.setdiff1d(pq, [bus])
						pv = np.sort(np.concatenate((pv, [bus])))
						self.q_min_bus = np.delete(self.q_min_bus, np.where(self.q_min_bus == bus))
						print("Not Q Limited: ", bus, " because ", v[bus], " < ",
							  self.bus_data[bus, self.busDesiredVolts])

		# Find buses that need to be limited.
		s_full = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
		q_calc_full = s_full.imag
		q_generated_full = q_calc_full + self.q_load_full
		q_min_limits = np.array([min(lim) for lim in q_lim])
		q_max_limits = np.array([max(lim) for lim in q_lim])
		# Keep record of all buses that are limited or have been limited.
		max_index_for_pv_buses = \
			np.where(np.array([max(lim) <= q_generated_full[i] for i, lim in enumerate(q_lim)])[pv])[0]
		min_index_for_pv_buses = \
			np.where(np.array([min(lim) >= q_generated_full[i] for i, lim in enumerate(q_lim)])[pv])[0]
		new_q_max_buses = np.array(pv[max_index_for_pv_buses])
		new_q_min_buses = np.array(pv[min_index_for_pv_buses])
		self.q_max_bus = np.unique(np.r_[self.q_max_bus, new_q_max_buses])
		self.q_min_bus = np.unique(np.r_[self.q_min_bus, new_q_min_buses])

		if self.q_min_bus.any() in pv or self.q_max_bus.any() in pv:
			if self.q_min_bus.any() in pv:  # Remove from pv list, add to pq list.
				pv = np.setdiff1d(pv, self.q_min_bus)
				pq = np.unique(np.concatenate((pq, self.q_min_bus)))
				self.qsched_full[self.q_min_bus] = \
					np.array(q_min_limits[self.q_min_bus] - self.q_load_full[self.q_min_bus])
			if self.q_max_bus.any() in pv:  # Remove from pv list, add to pq list.
				pv = np.setdiff1d(pv, self.q_max_bus)
				pq = np.unique(np.concatenate((pq, self.q_max_bus)))
				self.qsched_full[self.q_max_bus] = \
					np.array(q_max_limits[self.q_max_bus] - self.q_load_full[self.q_max_bus])
			q_limited = np.sort(np.concatenate((self.q_max_bus, self.q_min_bus)))
			print("Q Limited: ", q_limited)

		# Calculate scheduled Q for each bus
		qsched = self.qsched_full[pq]
		v[pv] = np.array(self.bus_data[pv, self.busDesiredVolts])

		return pv, pq, qsched

	def pf_jacobian(self, v, d, pq, decoupled=False, v_mul=True):
		# This function was written by Nathan Gray using formulas from chapter 9 of
		# "Power Systems Analysis" J. Grainger et al.
		# Calculates the Jacobian Matrix for use in the Newton-Raphson Method.
		# Arguments:
		# v: Voltage magnitudes
		# d: Voltage phase angles
		# y: Ybus matrix
		# pq: List of PQ buses
		y = self.y_bus
		n = y.shape[0]
		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
		s = (v*np.exp(1j*d))*np.conj(y.dot(v*np.exp(1j*d)))
		p = s.real
		q = s.imag

		if self.sparse:
			tmp = Sparse
		else:
			tmp = np

		# Find indices of non-zero ybus entries
		row, col = tmp.where(y)

		j11 = tmp.zeros((n - 1, n - 1))
		j12 = tmp.zeros((n - 1, pq.size))
		j21 = tmp.zeros((pq.size, n - 1))
		j22 = tmp.zeros((pq.size, pq.size))

		for a in range(row.shape[0]):
			i = row[a]
			j = col[a]
			# J11
			if i != 0 and j != 0:
				th_ij = np.angle(y[i, j])
				s_ij = np.sin(th_ij + d[j] - d[i])
				c_ij = np.cos(th_ij + d[j] - d[i])
				y_ij = abs(y[i, j])
				if i == j:  # Diagonals of J11
					j11[i - 1, j - 1] = - q[i] - v[i]**2*y[i, i].imag
				else:  # Off-diagonals of J11
					j11[i - 1, j - 1] = -v[i]*v[j]*y_ij*s_ij
				# J21
				if i in pq:
					k: int = np.ravel(np.where(pq == i))[0]  # map bus index to jacobian index
					if i == j:  # Diagonals of J21
						j21[k, j - 1] = p[i] - abs(v[i])**2*y[i, j].real
					else:  # Off-diagonals of J21
						j21[k, j - 1] = -v[i]*v[j]*y_ij*c_ij
				# J12
				if j in pq:
					l: int = np.ravel(np.where(pq == j))[0]  # map bus index to jacobian index
					if i == j:  # Diagonals of J12
						j12[i - 1, l] = p[i] + v[i]**2*y[i, j].real
					else:  # Off-diagonals of J12
						j12[i - 1, l] = v[i]*v[j]*y_ij*c_ij
					if not v_mul:
						j12[i - 1, l] /= v[j]
				# J22
				if i in pq and j in pq:
					k: int = np.ravel(np.where(pq == i))[0]  # map bus index to jacobian index
					l: int = np.ravel(np.where(pq == j))[0]  # map bus index to jacobian index
					if i == j:  # Diagonal of J22
						j22[k, l] = -j11[i - 1, j - 1] - 2*v[i]**2*y[i, j].imag
					else:  # Off-diagonals of J22
						j22[k, l] = j11[i - 1, j - 1]
					if not v_mul:
						j22[k, l] /= v[j]
		# Assemble jacobian
		jtop = tmp.concatenate((j11, j12), axis=1)
		jbottom = tmp.concatenate((j21, j22), axis=1)
		jacobian = tmp.concatenate((jtop, jbottom), axis=0)
		if decoupled:
			return j11, j22
		else:
			return jacobian

	def pf(self, initial=None, prec=5, maxit=10, qlim=False, qlim_prec=2, verbose=True, debug_file=None):
		if initial is None:
			v0, d0 = self.flat_start()
		else:
			v0, d0 = initial
		# d0 = self.pf_dc(d0, self.y_bus, self.pvpq, self.psched)
		v, d, it = self.pf_newtonraphson(v0, d0, prec=prec, maxit=maxit, qlim=qlim, qlim_prec=qlim_prec, verbose=verbose, debug_file=debug_file)
		return v, d

	def dslack_dx(self, v, d, pq):

		y = self.y_bus
		n = y.shape[0]
		if self.sparse:
			tmp = Sparse
			y = y.full[0, :]
		else:
			tmp = np
			y = y[0, :]
		# Find indices of non-zero ybus entries
		col = np.where(y)[0]

		j11 = np.zeros((1, n - 1))
		j12 = np.zeros((1, pq.size))

		for a in range(len(col)):
			i = 0
			j: int = col[a]
			if j != 0:
				th_ij = np.angle(y[j])
				s_ij = np.sin(th_ij + d[j])
				c_ij = np.cos(th_ij + d[j])
				y_ij = abs(y[j])

				j11[0, j - 1] = -v[i]*v[j]*y_ij*s_ij
				if j in pq:
					l: int = np.ravel(np.where(pq == j))[0]  # map bus index to jacobian index
					j12[0, l] = v[i]*v[j]*y_ij*c_ij/v[j]
		# Assemble jacobian
		dp1dx = np.concatenate((j11, j12), axis=1).T
		return dp1dx

	# ~~~~~ State Estimation ~~~~~
	def se_h_matrix(self, v, d):
		y = self.y_bus
		n = y.shape[0]
		nb = len(self.branch_data[:, 0])
		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
		s = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
		p = s.real
		q = s.imag

		# Find indices of non-zero ybus entries
		if self.sparse:
			row = y.rows
			col = y.cols
			tmp = Sparse
		else:
			row, col = np.where(y)
			tmp = np

		j01 = tmp.zeros((n, n - 1))
		j02 = tmp.zeros((n, n))
		j11 = tmp.zeros((n, n - 1))
		j12 = tmp.zeros((n, n))
		j21 = tmp.zeros((n, n - 1))
		j22 = tmp.zeros((n, n))
		j31 = tmp.zeros((nb, n - 1))
		j32 = tmp.zeros((nb, n))
		j41 = tmp.zeros((nb, n - 1))
		j42 = tmp.zeros((nb, n))
		j51 = tmp.zeros((nb, n - 1))
		j52 = tmp.zeros((nb, n))
		j61 = tmp.zeros((nb, n - 1))
		j62 = tmp.zeros((nb, n))

		for i in range(n):
			j02[i, i] = 1
		for a in range(row.shape[0]):
			i = row[a]
			j = col[a]
			th_ij = np.angle(y[i, j])
			# J11
			if j != 0:
				if i == j:  # Diagonals of J11  dPi/ddi
					j11[i, j - 1] = - q[i] - v[i] ** 2 * y[i, i].imag
				else:  # Off-diagonals of J11  dPi/ddj
					j11[i, j - 1] = -abs(v[i] * v[j] * y[i, j]) * np.sin(th_ij + d[j] - d[i])
				# J21
				if i == j:  # Diagonals of J21  dQi/ddi
					j21[i, j - 1] = p[i] - v[i] ** 2 * y[i, j].real
				else:  # Off-diagonals of J21  dQi/ddj
					j21[i, j - 1] = -abs(v[i] * v[j] * y[i, j]) * np.cos(th_ij + d[j] - d[i])
			# J12
			if i == j:  # Diagonals of J12  dPi/dvi
				j12[i, j] = (p[i] + abs(v[i] ** 2 * y[i, j].real)) / v[i]
			else:  # Off-diagonals of J12  dPi/dvj
				j12[i, j] = (abs(v[j] * v[i] * y[i, j]) * np.cos(th_ij + d[j] - d[i])) / v[j]
			# J22
			if i == j:  # Diagonal of J22  dQi/dvi
				j22[i, j] = (q[i] + v[i] ** 2 * y[i, i].imag - 2 * abs(v[i]) ** 2 * y[i, j].imag) / v[i]
			else:  # Off-diagonals of J22  dQi/dvj
				j22[i, j] = (-abs(v[i] * v[j] * y[i, j]) * np.sin(th_ij + d[j] - d[i])) / v[j]

		for b, _ in enumerate(self.branch_data[:, 0]):
			from_bus = self.branch_data[b, 0]
			to_bus = self.branch_data[b, 1]
			i = int(from_bus - 1)
			j = int(to_bus - 1)
			b_chrg = self.branch_data[b, self.branchB]
			ycosij = abs(y[i, j]) * cos(angle(y[i, j]) + d[j] - d[i])
			ycosji = abs(y[j, i]) * cos(angle(y[j, i]) + d[i] - d[j])
			ysinij = abs(y[i, j]) * sin(angle(y[i, j]) + d[j] - d[i])
			ysinji = abs(y[j, i]) * sin(angle(y[j, i]) + d[i] - d[j])

			if i != 0:  # Do not include derivatives w.r.t. d[0]
				# J31 dPij/dd[i]
				j31[b, i - 1] = v[i] * v[j] * ysinij
				# J41 dQij/dd[i]
				j41[b, i - 1] = v[i] * v[j] * ycosij
				# J51 dPji/dd[i]
				j51[b, i - 1] = -v[j] * v[i] * ysinji
				# J61 dQji/dd[i]
				j61[b, i - 1] = -v[j] * v[i] * ycosji

			if j != 0:  # Do not include derivatives w.r.t. d[0]
				# J31 dPij/dd[j]
				j31[b, j - 1] = -v[i] * v[j] * ysinij
				# J41 dQij/dd[j]
				j41[b, j - 1] = -v[i] * v[j] * ycosij
				# J51 dPji/dd[j]
				j51[b, j - 1] = v[j] * v[i] * ysinji
				# J61 dQji/dd[j]
				j61[b, j - 1] = v[j] * v[i] * ycosji

			# J32 dPij/dV
			j32[b, i] = -2 * v[i] * real(y[i, j]) + v[j] * ycosij
			j32[b, j] = v[i] * ycosij
			# J42 dQij/dV
			j42[b, i] = -2 * v[i] * (b_chrg / 2 - imag(y[i, j])) - v[j] * ysinij
			j42[b, j] = -v[i] * ysinij
			# J52 dPji/dV
			j52[b, j] = -2 * v[j] * real(y[j, i]) + v[i] * ycosji
			j52[b, i] = v[j] * ycosji
			# J62 dQji/dV
			j62[b, j] = -2 * v[j] * (b_chrg / 2 - imag(y[j, i])) - v[i] * ysinji
			j62[b, i] = -v[j] * ysinji

		# Assemble jacobian
		if self.sparse:
			j0 = Sparse.concatenate((j01, j02), axis=1)
			j1 = Sparse.concatenate((j11, j12), axis=1)
			j2 = Sparse.concatenate((j21, j22), axis=1)
			j3 = Sparse.concatenate((j31, j32), axis=1)
			j4 = Sparse.concatenate((j41, j42), axis=1)
			j5 = Sparse.concatenate((j51, j52), axis=1)
			j6 = Sparse.concatenate((j61, j62), axis=1)
			j0_1 = Sparse.concatenate((j0, j1), axis=0)
			j2_3 = Sparse.concatenate((j2, j3), axis=0)
			j4_5 = Sparse.concatenate((j4, j5), axis=0)
			j0_3 = Sparse.concatenate((j0_1, j2_3), axis=0)
			j4_6 = Sparse.concatenate((j4_5, j6), axis=0)
			jacobian = Sparse.concatenate((j0_3, j4_6), axis=0)
		else:
			j0 = np.concatenate((j01, j02), axis=1)
			j1 = np.concatenate((j11, j12), axis=1)
			j2 = np.concatenate((j21, j22), axis=1)
			j3 = np.concatenate((j31, j32), axis=1)
			j4 = np.concatenate((j41, j42), axis=1)
			j5 = np.concatenate((j51, j52), axis=1)
			j6 = np.concatenate((j61, j62), axis=1)
			jacobian = np.concatenate((j0, j1, j2, j3, j4, j5, j6), axis=0)

		return jacobian

	def h_calc(self, v, d):
		v = np.transpose(v)[0]
		d = np.transpose(d)[0]
		s = (v * np.exp(1j * d)) * np.conj(self.y_bus.dot(v * np.exp(1j * d)))
		p = np.real(s)
		q = np.imag(s)
		pij, qij, pji, qji = self.branch_flows(v, d)
		return np.r_[
			np.transpose([v]),
			np.transpose([p]),
			np.transpose([q]),
			np.transpose([pij]),
			np.transpose([qij]),
			np.transpose([pji]),
			np.transpose([qji])
		]

	@staticmethod
	def mismatch(v, d, y, pq, pvpq, psched, qsched):
		# This function was written by Nathan Gray
		# This function calculates mismatches between the real and reactive power
		# injections in a system vs. the scheduled injections.
		# power system network.
		# Arguments:
		# v: list of voltage magnitudes in system
		# d: list of voltage phase angles in system
		# y: Ybus matrix for system
		# pq: list of PQ buses
		# pvpq: list of PV and pq buses
		# psched, qsched: list of real, reactive power injections

		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
		s = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
		# S = P + jQ
		pcalc = s[pvpq].real
		qcalc = s[pq].imag
		dp = psched - pcalc
		dq = qsched - qcalc
		mis = np.concatenate((dp, dq))
		return mis, pcalc, qcalc

	def complex_injections(self, v, d):
		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
		s = (v * np.exp(1j * d)) * np.conj(self.y_bus.dot(v * np.exp(1j * d)))
		return s

	def pij_flow(self, d, v, i, j):
		yij = np.abs(self.y_bus[i, j])
		gij = np.real(self.y_bus[i, j])
		th_ij = np.angle(self.y_bus[i, j])
		p_ij = -v[i] ** 2 * gij + v[i] * v[j] * yij * np.cos(th_ij + d[j] - d[i])
		return p_ij

	def qij_flow(self, d, v, i, j, b):
		b_charging = self.branch_data[b, self.branchB]
		yij = np.abs(self.y_bus[i, j])
		bij = np.imag(self.y_bus[i, j])
		th_ij = np.angle(self.y_bus[i, j])
		q_ij = -v[i] ** 2 * (b_charging / 2 - bij) - v[i] * v[j] * yij * np.sin(th_ij + d[j] - d[i])
		return q_ij

	def branch_flows(self, v, d):
		p_ij = np.zeros(self.branch_data[:, 0].shape)
		p_ji = np.zeros(self.branch_data[:, 0].shape)
		q_ij = np.zeros(self.branch_data[:, 0].shape)
		q_ji = np.zeros(self.branch_data[:, 0].shape)
		b_charging = self.branch_data[:, self.branchB]
		for b, _ in enumerate(self.branch_data[:, 0]):
			from_bus = self.branch_data[b, 0]
			to_bus = self.branch_data[b, 1]
			i = int(from_bus - 1)
			j = int(to_bus - 1)
			p_ij[b] = self.pij_flow(d, v, i, j)
			q_ij[b] = self.qij_flow(d, v, i, j, b)

			p_ji[b] = self.pij_flow(d, v, j, i)
			q_ji[b] = self.qij_flow(d, v, j, i, b)
		return p_ij, q_ij, p_ji, q_ji

	# ~~~~~ Continuation Power flow aka Voltage Stability Analyis ~~~~~
	def cpf_jacobian(self, v, d, pq, kpq, kt, sign):
		# Build parameterized jacobian for continuation power flow.
		y = self.y_bus
		n = y.shape[0]
		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
		s = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
		p = s.real
		q = s.imag
		if self.sparse:
			tmp = Sparse
		else:
			tmp = np
		# Find indices of non-zero ybus entries
		row, col = tmp.where(y)
		j11 = tmp.zeros((n - 1, n - 1))
		j12 = tmp.zeros((n - 1, pq.size))
		j21 = tmp.zeros((pq.size, n - 1))
		j22 = tmp.zeros((pq.size, pq.size))
		for a in range(row.shape[0]):
			i = row[a]
			j = col[a]
			# J11
			if i != 0 and j != 0:
				th_ij = np.angle(y[i, j])
				s_ij = np.sin(th_ij + d[j] - d[i])
				c_ij = np.cos(th_ij + d[j] - d[i])
				y_ij = abs(y[i, j])
				if i == j:  # Diagonals of J11
					j11[i - 1, j - 1] = -q[i] - v[i]**2*y[i, i].imag
				else:  # Off-diagonals of J11
					j11[i - 1, j - 1] = -v[i]*v[j]*y_ij*s_ij
				# J21
				if i in pq:
					k: int = np.ravel(np.where(pq == i))[0]  # map bus index to jacobian index
					if i == j:  # Diagonals of J21
						j21[k, j - 1] = p[i] - v[i]**2*y[i, j].real
					else:  # Off-diagonals of J21
						j21[k, j - 1] = -v[i]*v[j]*y_ij*c_ij
				# J12
				if j in pq:
					l: int = np.ravel(np.where(pq == j))[0]  # map bus index to jacobian index
					if i == j:  # Diagonals of J12
						j12[i - 1, l] = (p[i] + v[i]**2*y[i, j].real) / v[j]
					else:  # Off-diagonals of J12
						j12[i - 1, l] = (v[i]*v[j]*y_ij*c_ij) / v[j]
				# J22
				if i in pq and j in pq:
					k: int = np.ravel(np.where(pq == i))[0]  # map bus index to jacobian index
					l: int = np.ravel(np.where(pq == j))[0]  # map bus index to jacobian index
					if i == j:  # Diagonal of J22
						j22[k, l] = (-j11[i - 1, j - 1] - 2*v[i]**2*y[i, j].imag) / v[j]
					else:  # Off-diagonals of J22
						j22[k, l] = j11[i - 1, j - 1] / v[j]
		# Assemble jacobian
		jtop = tmp.concatenate((j11, j12), axis=1)
		jbottom = tmp.concatenate((j21, j22), axis=1)
		jacobian = tmp.concatenate((jtop, jbottom), axis=0)

		jac = jacobian
		# add row for ek and col for psched and qsched
		nrows = jac.shape[0]
		ncols = jac.shape[1]
		if self.sparse:
			for row in range(nrows):
				jac[row, ncols] = -kpq[row]
			jac[nrows, kt] = sign
		else:
			ek = np.zeros((1, ncols))
			ek[kt] = sign
			jac = np.c_[jac, -kpq]
			jac = np.r_[jac, ek]
		return jac

	def pf_continuation(self, watch_bus):
		print("\n~~~~~~~~~~ Start Voltage Stability Analysis ~~~~~~~~~~\n")
		σ = 0.1
		λ = 1
		psched = deepcopy(self.psched)
		qsched = deepcopy(self.qsched)
		kpq = np.r_[psched, qsched]
		y = self.y_bus
		n = np.shape(y)[0]
		pvpq = self.pvpq
		pq = deepcopy(self.pq)
		# ~~~~~~~ Run Conventional Power Flow on Base Case ~~~~~~~~~~
		v, d = self.flat_start()
		d = self.pf_dc(d, y, pvpq, psched, lam=λ)
		v, d, it = self.pf_newtonraphson(v, d, prec=3, maxit=10, qlim=False, lam=λ)
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# ~~~~~ Set watched bus and associated indexes ~~~~~
		# watch_bus = 4
		watch_index = watch_bus - 1
		watch_pq_index = watch_index  # initialize
		for i, bus_type in enumerate(self.bus_data[:, self.busType]):
			if watch_index <= i:
				break
			if bus_type > 0 and watch_index > i:
				watch_pq_index -= 1
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		results = [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]
		phase = 1  # phase 1 -> increasing load, phase 2 -> decreasing V, phase 3 -> decreasing load

		# Continuation Power Flow or Voltage Stability Analysis
		while True:

			# kpq_jon = np.zeros(kpq.shape)
			# kpq_jon[watch_index-1] = -1
			# Calculate Jacobian
			if phase == 1:
				kt = len(pvpq) + len(pq)
				tk = 1
				jac = self.cpf_jacobian(v, d, pq, kpq, kt, tk)
			if phase == 2:
				kt = len(pvpq) + watch_pq_index
				tk = -1
				jac = self.cpf_jacobian(v, d, pq, kpq, kt, tk)
			if phase == 3:
				kt = len(pvpq) + len(pq)
				tk = -1
				jac = self.cpf_jacobian(v, d, pq, kpq, kt, tk)

			# Calculate update values
			# ~~~~~~~~~~ Calculated Tangent Vector ~~~~~~~~~~
			t = mat_solve(jac, np.r_[np.zeros(jac.shape[0] - 1), 1])
			# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			# Update angles: d_(n+1) = d_n + dd
			d_pred = deepcopy(d)
			d_pred[pvpq] = d[pvpq] + σ * t[:n - 1]
			# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
			v_pred = deepcopy(v)
			v_pred[pq] = v[pq] + σ * t[n - 1:-1]
			# Update Lambda
			λ_pred = λ + σ * t[-1]
			# ~~~~~~~~~~ Corrector ~~~~~~~~~~

			d_cor = deepcopy(d_pred)
			v_cor = deepcopy(v_pred)
			λ_cor = deepcopy(λ_pred)
			it = 0
			maxit = 7
			while it < maxit:
				mis, p_calc, q_calc = self.mismatch(v_cor, d_cor, y, pq, pvpq, λ_cor*psched, λ_cor*qsched)
				if phase == 1 or phase == 3:
					mis = np.r_[mis, λ_pred - λ_cor]
				if phase == 2:
					mis = np.r_[mis, v_pred[watch_index] - v_cor[watch_index]]
				# Check error
				if max(abs(mis)) < 10 ** -3:
					break  # return v, d, it
				jac = self.cpf_jacobian(v_cor, d_cor, pq, kpq, kt, tk)
				# Calculate update values
				dx = mat_solve(jac, mis)
				# Update angles: d_(n+1) = d_n + dd
				d_cor[pvpq] = d_cor[pvpq] + dx[:n - 1]
				# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
				v_cor[pq] = v_cor[pq] + dx[n - 1:n + pq.size - 1]
				# Update Lambda
				λ_cor = λ_cor + dx[-1]
				it += 1

			if phase == 1:
				if it >= maxit:
					phase = 2
					σ = 0.025
					print('phase 2')
				else:
					v = deepcopy(v_cor)
					d = deepcopy(d_cor)
					λ = deepcopy(λ_cor)
					print(round(λ, 8), v[watch_index])
					results = np.r_[results, [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]]

			elif phase == 2:
				if it >= maxit:
					print("phase 2 not converged")
					#break
					phase = 3
					σ = 0.1
					print('phase 3')
				elif results[-2, 3] - results[-1, 3] > 0.2:
					phase = 3
					σ = 0.1
					print('phase 3')
				else:
					v = deepcopy(v_cor)
					d = deepcopy(d_cor)
					λ = deepcopy(λ_cor)
					print(round(λ, 8), v[watch_index])
					results = np.r_[results, [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]]

			if phase == 3:
				# break
				if λ < 1:
					break

				v = deepcopy(v_cor)
				d = deepcopy(d_cor)
				λ = deepcopy(λ_cor)
				print(round(λ, 8), v[watch_index])
				results = np.r_[results, [[σ, v[watch_index], d[watch_index], λ, λ*self.psched[watch_index - 1], λ*self.qsched[watch_pq_index ]]]]

		return results

	def dgdx(self, x):
		lenx = len(x)
		d = np.r_[0, x[0:lenx//2]]
		v = np.r_[self.bus_data[0, self.busDesiredVolts], x[lenx//2:]]
		y = self.y_bus
		n = y.shape[0]
		nb = len(self.branch_data[:, 0])
		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
		s = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
		p = s.real
		q = s.imag

		# Find indices of non-zero ybus entries
		if self.sparse:
			row = y.rows
			col = y.cols
			tmp = Sparse
		else:
			row, col = np.where(y)
			tmp = np

		j11 = tmp.zeros((n - 1, n - 1))
		j12 = tmp.zeros((n - 1, n - 1))
		j21 = tmp.zeros((n - 1, n - 1))
		j22 = tmp.zeros((n - 1, n - 1))

		for a in range(row.shape[0]):
			i = row[a]
			j = col[a]
			th_ij = np.angle(y[i, j])

			if i != 0 and j != 0:  # J11
				if i == j:  # Diagonals of J11  dPi/ddi
					j11[i - 1, j - 1] = - q[i] - v[i] ** 2 * y[i, i].imag
				else:  # Off-diagonals of J11  dPi/ddj
					j11[i - 1, j - 1] = -abs(v[i] * v[j] * y[i, j]) * np.sin(th_ij + d[j] - d[i])
			# if j != 0:  # J21
				if i == j:  # Diagonals of J21  dQi/ddi
					j21[i - 1, j - 1] = p[i] - v[i] ** 2 * y[i, j].real
				else:  # Off-diagonals of J21  dQi/ddj
					j21[i - 1, j - 1] = -abs(v[i] * v[j] * y[i, j]) * np.cos(th_ij + d[j] - d[i])
			# if i != 0:  # J12
				if i == j:  # Diagonals of J12
					j12[i - 1, j - 1] = (p[i] + abs(v[i] ** 2 * y[i, j].real)) / v[i]
				else:  # Off-diagonals of J12
					j12[i - 1, j - 1] = (abs(v[j] * v[i] * y[i, j]) * np.cos(th_ij + d[j] - d[i])) / v[j]
			# J22
				if i == j:  # Diagonal of J22
					j22[i - 1, j - 1] = (q[i] + v[i] ** 2 * y[i, i].imag - 2 * abs(v[i]) ** 2 * y[i, j].imag) / v[i]
				else:  # Off-diagonals of J22
					j22[i - 1, j - 1] = (-abs(v[i] * v[j] * y[i, j]) * np.sin(th_ij + d[j] - d[i])) / v[j]

		# Assemble jacobian
		j1 = tmp.concatenate((j11, j12), axis=1)
		j2 = tmp.concatenate((j21, j22), axis=1)
		jacobian = tmp.concatenate((j1, j2), axis=0)

		return jacobian

	def g(self, x):
		lenx = len(x)
		d = np.r_[0, x[0:lenx//2]]
		v = np.r_[self.bus_data[0, self.busDesiredVolts], x[lenx//2:]]
		s = self.complex_injections(v, d)
		p = s.real
		q = s.imag
		return np.r_[self.psched_full[1:] - p[1:], self.qsched_full[1:] - q[1:]]

	def nr(self, func, x0, fprime, maxit=10, prec=3, verbose=True):
		x = deepcopy(x0)
		for i in range(maxit + 1):
			# Calculate Mismatches
			mis = func(x)
			if verbose:
				print("error: ", max(abs(mis)))
			# Check error
			if max(abs(mis)) < 10 ** -abs(prec):
				if verbose:
					print("Newton Raphson completed in ", i, " iterations.")
				return x, i
			# Calculate Jacobian
			j = fprime(x)
			# Calculate update values
			dx = mat_solve(j, mis)
			# Update angles: d_(n+1) = d_n + dd
			x = x - dx

	def diff(self, func, x_eq):
		mat = np.zeros((len(x_eq), len(x_eq)))
		dx = np.zeros(len(x_eq))
		h = 1e-8
		for i in range(len(x_eq)):
			for j in range(len(x_eq)):
				dx[j] = h
				mat[i, j] = (func(x_eq + dx / 2)[i] - func(x_eq - dx / 2)[i]) / h
				dx[j] = 0
		return mat

if __name__ == "__main__":
	import matplotlib.pyplot as plt

	# case_name = "IEEE14BUS.txt"
	case_name = "IEEE14BUS_handout.txt"
	# case_name = "2BUS.txt"
	ps = PowerSystem(case_name, sparse=True)
	# v0, d0 = ps.flat_start()
	# v_nr, d_nr, it = ps.pf_newtonraphson(v0, d0, prec=2, maxit=10, qlim=False, lam=4)
	watch_bus = 14
	results = ps.pf_continuation(watch_bus)
	nose_point_index = np.argmax(results[:, 3])
	nose_point = results[nose_point_index, :]
	print(nose_point)
	plt.plot(results[:, 3], results[:, 1], '-o')
	plt.title('PV Curve for Modified IEEE 14-Bus System at Bus {}'.format(watch_bus))
	plt.xlabel('Lambda (schedule multiplication factor)')
	plt.ylabel('Bus Voltage (p.u.)')
	plt.show()
