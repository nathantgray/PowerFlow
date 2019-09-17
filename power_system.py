import numpy as np
from copy import deepcopy
from crout import mat_solve


class PowerSystem:
	def __init__(self, filename):
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
		# slack = np.where(types == 3)
		self.pv = np.where(types == 2)[0]  # list of PV bus indices
		self.pq = np.where(types < 2)[0]  # list of PQ bus indices
		self.pvpq = np.sort(np.concatenate((self.pv, self.pq)))  # list of indices of non-slack buses

		# Calculate scheduled P and Q for each bus
		self.mw_gen = self.bus_data[self.pvpq, self.busGenMW]
		self.mw_load = self.bus_data[self.pvpq, self.busLoadMW]
		self.mvar_load = self.bus_data[self.pq, self.busLoadMVAR]
		self.psched = np.array(self.mw_gen - self.mw_load)/self.p_base
		self.qsched = np.array(- self.mvar_load)/self.p_base
		self.q_lim = np.array([self.bus_data[:, self.busMaxMVAR], self.bus_data[:, self.busMinMVAR]]).transpose()/self.p_base

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

	def makeybus(self):
		# Produces the Y bus matrix of a power system.
		# Written by Nathan Gray
		# Arguments:
		# bus_data: Bus data from the IEEE common data format as a numpy array
		# branch_data: Branch data from the IEEE common data format as a numpy array

		busShuntG = 15
		busShuntB = 16

		branchR = 6
		branchX = 7
		branchB = 8
		branchTurnsRatio = 14
		branchPhaseShift = 15

		# Prepare data for algorithm
		z = self.branch_data[:, branchR] + self.branch_data[:, branchX] * 1j
		y = z ** -1
		b_line = self.branch_data[:, branchB]
		ratio = np.where(self.branch_data[:, branchTurnsRatio] == 0.0, 1, self.branch_data[:, branchTurnsRatio])
		shift = np.radians(self.branch_data[:, branchPhaseShift])
		t = ratio * np.cos(shift) + 1j * ratio * np.sin(shift)
		# Shunt admittances for each bus.
		y_shunt = self.bus_data[:, busShuntG] + 1j * self.bus_data[:, busShuntB]
		frombus = self.branch_data[:, 0]
		tobus = self.branch_data[:, 1]

		nl = self.branch_data.shape[0]  # number of lines
		n = self.bus_data.shape[0]  # number of buses
		y_bus = np.zeros((n, n)) + np.zeros((n, n)) * 1j  # initialize Y Bus Matrix

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
		for i in range(n):
			y_bus[i, i] += y_shunt[i]

		return y_bus

	def pf_newtonraphson(self, v_start, d_start, prec=2, maxit=4):
		# Uses Newton-Raphson method to solve the power-flow of a power system.
		# Written by Nathan Gray
		# Arguments:
		# v: list of voltage magnitudes in system
		# d: list of voltage phase angles in system
		# y: Ybus matrix for system
		# pq: list of PQ buses
		# pv: list of PV buses
		# psched, qsched: list of real, reactive power injections
		# qlim: [qmax, qmin]
		# prec: program finishes when all mismatches < 10^-abs(prec)

		psched = self.psched
		qsched = deepcopy(self.qsched)
		v = deepcopy(v_start)
		d = deepcopy(d_start)
		y = self.y_bus
		pvpq = self.pvpq
		pq = deepcopy(self.pq)
		pv = deepcopy(self.pv)
		pq_last = deepcopy(pq)
		n = np.shape(y)[0]
		# Newton Raphson
		it = 0
		for it in range(maxit+1):
			# Calculate Mismatches
			if it > 0:
				pq_last = deepcopy(pq)
				pv, pq, qsched = self.check_limits(v, d, y, pv, pq, qsched)
			mis, p_calc, q_calc = self.mismatch(v, d, y, pq, pvpq, psched, qsched)
			print("error: ", max(mis))
			print(q_calc)
			# Check error
			if max(abs(mis)) < 10**-abs(prec) and pq_last.all() == pq.all():
				print("Newton Raphson completed in ", it, " iterations.")
				return v, d, it
			# Calculate Jacobian
			j = self.pf_jacobian(v, d, pq)
			# Calculate update values
			dx = mat_solve(j, mis)
			# Update angles: d_(n+1) = d_n + dd
			d[pvpq] = d[pvpq] + dx[:n - 1]
			# Update Voltages: V_(n+1) = V_n(1+dV/V_n)
			v[pq] = v[pq]*(1+dx[n-1:n+pq.size-1])

		print("Max iterations reached, ", it, ".")
		return v, d, it

	def pf_fast_decoupled(self, v_start, d_start, prec=2, maxit=100):
		print("\n~~~~~~~~~~ Start Fast Decoupled Method ~~~~~~~~~~\n")
		psched = self.psched
		qsched = deepcopy(self.qsched)
		y = self.y_bus
		v = deepcopy(v_start)
		d = deepcopy(d_start)
		pvpq = self.pvpq
		pq = self.pq
		pv = self.pv
		pq_last = deepcopy(pq)
		# Decoupled Power Flow
		it = 0
		# bd_ = np.zeros(y.shape)
		# bd_[np.where(y)] = 1/(np.imag(1/y[np.where(y)]))
		# bd_ = bd_[pvpq, :][:, pvpq]
		bd = -y.imag[pvpq, :][:, pvpq]
		bv = -y.imag[pq, :][:, pq]
		for it in range(maxit+1):
			# Do q-limit check after first iteration
			if it > 0:
				pq_last = deepcopy(pq)
				pv, pq, qsched = self.check_limits(v, d, y, pv, pq, qsched)
			# Only update bv matrix size if pq changes
			if not np.array_equiv(pq_last, pq):
				bv = -y.imag[pq, :][:, pq]
			# Calculate Mismatches
			mis, p_calc, q_calc = self.mismatch(v, d, y, pq, pvpq, psched, qsched)
			print("error: ", max(mis))
			print(q_calc)
			# Check error
			if max(abs(mis)) < 10 ** -abs(prec) and pq_last.all() == pq.all():
				print("Decoupled Power Flow completed in ", it, " iterations.")
				return v, d, it
			d[pvpq] = d[pvpq] + mat_solve(bd, mis[0:len(pvpq)] / v[pvpq])
			v[pq] = v[pq] + mat_solve(bv, mis[len(pvpq):] / v[pq])

		print("Max iterations reached, ", it, ".")
		return v, d, it

	def check_limits(self, v, d, y, pv, pq, qsched):
		q_lim = self.q_lim
		# pq = deepcopy(self.pq)
		# pv = deepcopy(self.pv)
		# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
		s = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
		q_calc = s.imag
		q_gen = q_calc + np.array(self.bus_data[:, self.busLoadMVAR])/self.p_base
		g_bus = deepcopy(pv)
		q_min = np.array([min(lim) for lim in q_lim])
		q_max = np.array([max(lim) for lim in q_lim])
		# Get lists of non-slack generator buses that are limited by max and min limits
		# q_max_bus = np.array(g_bus[np.where(np.array([max(lim) <= q_gen[i] for i, lim in enumerate(q_lim)])[g_bus])[0]])
		# q_min_bus = np.array(g_bus[np.where(np.array([min(lim) >= q_gen[i] for i, lim in enumerate(q_lim)])[g_bus])[0]])

		if q_min_bus.any() in pv or q_max_bus.any() in pv:
			q_gen_set = np.zeros(q_calc.shape)
			if q_min_bus.any() in pv:  # Remove from pv list, add to pq list.
				pv = np.setdiff1d(pv, q_min_bus)
				pq = np.sort(np.concatenate((pq, q_min_bus)))
				q_gen_set[q_min_bus] = q_min[q_min_bus]
			if q_max_bus.any() in pv:  # Remove from pv list, add to pq list.
				pv = np.setdiff1d(pv, q_max_bus)
				pq = np.sort(np.concatenate((pq, q_max_bus)))
				q_gen_set[q_max_bus] = q_max[q_max_bus]
			q_limited = np.sort(np.concatenate((q_max_bus, q_min_bus)))
			q_load_pu = np.array(self.bus_data[pq, self.busLoadMVAR] / self.p_base)
			qsched = np.array(q_gen_set[pq] - q_load_pu)
			print("Q Limited: ", q_limited)
		# Calculate scheduled Q for each bus
		v[pv] = np.array(self.bus_data[:, self.busDesiredVolts])[pv]

		return pv, pq, qsched

	def pf_jacobian(self, v, d, pq):
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
		s = (v * np.exp(1j * d)) * np.conj(y.dot(v * np.exp(1j * d)))
		p = s.real
		q = s.imag

		# Find indices of non-zero ybus entries
		row, col = np.where(y)

		j11 = np.zeros((n - 1, n - 1))
		j12 = np.zeros((n - 1, pq.size))
		j21 = np.zeros((pq.size, n - 1))
		j22 = np.zeros((pq.size, pq.size))
		for a in range(row.shape[0]):
			i = row[a]
			j = col[a]
			# J11
			if i != 0 and j != 0:
				if i == j:  # Diagonals of J11
					j11[i - 1, j - 1] = - q[i] - v[i] ** 2 * y[i, i].imag
				else:  # Off-diagonals of J11
					j11[i - 1, j - 1] = -abs(v[i] * v[j] * y[i, j]) * np.sin(np.angle(y[i, j]) + d[j] - d[i])
				# J21
				if i in pq:
					k: int = np.where(pq == i)  # map bus index to jacobian index
					if i == j:  # Diagonals of J21
						j21[k, j - 1] = p[i] - abs(v[i]) ** 2 * y[i, j].real
					else:  # Off-diagonals of J21
						j21[k, j - 1] = -abs(v[i] * v[j] * y[i, j]) * np.cos(np.angle(y[i, j]) + d[j] - d[i])
				# J12
				if j in pq:
					l: int = np.where(pq == j)  # map bus index to jacobian index
					if i == j:  # Diagonals of J12
						j12[i - 1, l] = p[i] + abs(v[i] ** 2 * y[i, j].real)
					else:  # Off-diagonals of J12
						j12[i - 1, l] = abs(v[j] * v[i] * y[i, j]) * np.cos(np.angle(y[i, j]) + d[j] - d[i])
				# J22
				if i in pq and j in pq:
					k: int = np.where(pq == i)  # map bus index to jacobian index
					l: int = np.where(pq == j)  # map bus index to jacobian index
					if i == j:  # Diagonal of J22
						j22[k, l] = -j11[i - 1, j - 1] - 2 * abs(v[i]) ** 2 * y[i, j].imag
					else:  # Off-diagonals of J22
						j22[k, l] = j11[i - 1, j - 1]
		# Assemble jacobian
		jtop = np.concatenate((j11, j12), axis=1)
		jbottom = np.concatenate((j21, j22), axis=1)
		jacobian = np.concatenate((jtop, jbottom), axis=0)
		return jacobian

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

	def flat_start(self):
		# Initialize with flat start
		v_flat = np.array(
			np.where(self.bus_data[:, self.busDesiredVolts] == 0.0, 1, self.bus_data[:, self.busDesiredVolts]))
		d_flat = np.zeros(v_flat.shape)
		return v_flat, d_flat


if __name__ == "__main__":
	# case_name = "IEEE14BUS.txt"
	case_name = "IEEE14BUS_handout.txt"
	ps = PowerSystem(case_name)
	v0, d0 = ps.flat_start()
	v, d, it = ps.pf_newtonraphson(v0, d0, prec=3, maxit=10)
	# v, d, it = ps.pf_fast_decoupled(v0, d0, prec=2, maxit=40)
	s = (v * np.exp(1j * d)) * np.conj(ps.y_bus.dot(v * np.exp(1j * d)))
	print(v)
	print(s.real*ps.p_base)
	print(s.imag*ps.p_base)