import numpy as np
import pandas as pd
from copy import deepcopy
from classes.power_system import PowerSystem
from classes.crout_cuda import mat_solve
from classes.sparse import Sparse
from numpy import sin, cos, angle, imag, real
from numba import njit

@njit
def _makeybus(make_bpp=False, make_bp=False, override=None):
	# Produces the Y bus matrix of a power system.
	# Written by Nathan Gray
	bus_data = override[0]
	branch_data = override[1]
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

	# if self.sparse:
	# 	y_bus = Sparse.zeros((n, n), dtype=complex)  # initialize Y Bus Matrix
	# else:
	y_bus = np.zeros((n, n), dtype=np.complex64)  # initialize Y Bus Matrix
	# The following algorithm takes the arguments: y, b_line, t, y_shunt
	# Create the four entries of a Y-Bus matrix for each line.
	#
	# i|-}|{--~~--|j
	# 	 t:1   y
	#
	# [y/|t|^2   -y/t*]
	# [-y/t      y  ]

	yjj = y + 1j * b_line / 2
	yii = yjj / (np.abs(t) ** 2)
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


@njit
def _pf_jacobian(y_bus, v, d, pq, v_mul=True):
	# This function was written by Nathan Gray using formulas from chapter 9 of
	# "Power Systems Analysis" J. Grainger et al.
	# Calculates the Jacobian Matrix for use in the Newton-Raphson Method.
	# Arguments:
	# v: Voltage magnitudes
	# d: Voltage phase angles
	# y: Ybus matrix
	# pq: List of PQ buses
	y = y_bus
	n = y.shape[0]
	# S = V*conj(I) and I = Y*V => S = V*conj(Y*V)
	s = (v * np.exp(1j * d)) * np.conj(np.dot(y, (v * np.exp(1j * d))))
	p = s.real
	q = s.imag

	# if sparse:
	# 	tmp = Sparse
	# else:
	# 	tmp = np
	tmp = np
	# Find indices of non-zero ybus entries
	row, col = tmp.where(y)

	j11 = tmp.zeros((n - 1, n - 1))
	j12 = tmp.zeros((n - 1, pq.size))
	j21 = tmp.zeros((pq.size, n - 1))
	j22 = tmp.zeros((pq.size, pq.size))
	# pq = list(pq)
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
				j11[i - 1, j - 1] = - q[i] - v[i] ** 2 * y[i, i].imag
			else:  # Off-diagonals of J11
				j11[i - 1, j - 1] = -v[i] * v[j] * y_ij * s_ij
			# J21
			if i in list(pq):
				k: int = np.where(pq == i)[0][0]  # map bus index to jacobian index
				if i == j:  # Diagonals of J21
					j21[k, j - 1] = p[i] - abs(v[i]) ** 2 * y[i, j].real
				else:  # Off-diagonals of J21
					j21[k, j - 1] = -v[i] * v[j] * y_ij * c_ij
			# J12
			if j in list(pq):
				l: int = np.where(pq == j)[0][0]  # map bus index to jacobian index
				if i == j:  # Diagonals of J12
					j12[i - 1, l] = p[i] + v[i] ** 2 * y[i, j].real
				else:  # Off-diagonals of J12
					j12[i - 1, l] = v[i] * v[j] * y_ij * c_ij
				if not v_mul:
					j12[i - 1, l] /= v[j]
			# J22
			if i in list(pq) and j in list(pq):
				k: int = np.where(pq == i)[0][0]  # map bus index to jacobian index
				l: int = np.where(pq == j)[0][0]  # map bus index to jacobian index
				if i == j:  # Diagonal of J22
					j22[k, l] = -j11[i - 1, j - 1] - 2 * v[i] ** 2 * y[i, j].imag
				else:  # Off-diagonals of J22
					j22[k, l] = j11[i - 1, j - 1]
				if not v_mul:
					j22[k, l] /= v[j]
	# Assemble jacobian
	jtop = tmp.concatenate((j11, j12), axis=1)
	jbottom = tmp.concatenate((j21, j22), axis=1)
	jacobian = tmp.concatenate((jtop, jbottom), axis=0)
	# if decoupled:
	# 	return j11, j22
	# else:
	return jacobian

class PowerSystem_CUDA(PowerSystem):
	def __init__(self, filename, sparse=False):
		PowerSystem.__init__(self, filename, sparse=False)

	def makeybus(self, make_bpp=False, make_bp=False, override=None):
		if not isinstance(override, tuple):
			bus_data = self.bus_data
			branch_data = self.branch_data
			override = (bus_data, branch_data)
		return _makeybus(make_bpp=make_bpp, make_bp=make_bp, override=override)

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
			j = self.pf_jacobian_cuda(v, d, pq)
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

	def pf_jacobian_cuda(self, v, d, pq, decoupled=False, v_mul=True):
		return _pf_jacobian(self.y_bus, v, d, pq, v_mul=v_mul)

	@staticmethod
	@njit
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
		s = (v * np.exp(1j * d)) * np.conj(np.dot(y, v * np.exp(1j * d)))
		# S = P + jQ
		pcalc = s[pvpq].real
		qcalc = s[pq].imag
		dp = psched - pcalc
		dq = qsched - qcalc
		mis = np.concatenate((dp, dq))
		return mis, pcalc, qcalc

	def pf(self, initial=None, prec=5, maxit=10, qlim=False, qlim_prec=2, verbose=True, debug_file=None):
		if initial is None:
			v0, d0 = self.flat_start()
		else:
			v0, d0 = initial
		# d0 = self.pf_dc(d0, self.y_bus, self.pvpq, self.psched)
		v, d, it = self.pf_newtonraphson(v0, d0, prec=prec, maxit=maxit, qlim=qlim, qlim_prec=qlim_prec, verbose=verbose, debug_file=debug_file)
		return v, d

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
