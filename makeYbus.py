import numpy as np


def makeybus(bus_data, branch_data):
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

	z = branch_data[:, branchR] + branch_data[:, branchX] * 1j
	y = z**-1
	b_line = branch_data[:, branchB]
	ratio = np.where(branch_data[:, branchTurnsRatio] == 0.0, 1, branch_data[:, branchTurnsRatio])
	shift = np.radians(branch_data[:, branchPhaseShift])
	t = ratio*np.cos(shift) + 1j*ratio*np.sin(shift)
	# Shunt admittances for each bus.
	y_shunt = bus_data[:, busShuntG] + 1j * bus_data[:, busShuntB]
	frombus = int(branch_data[:, 0])
	tobus = int(branch_data[:, 1])

	nl = branch_data.shape[0]  # number of lines
	n = bus_data.shape[0]  # number of buses
	y_bus = np.zeros((n, n)) + np.zeros((n, n))*1j  # initialize Y Bus Matrix

	# The following algorithm takes the arguments: y, b_line, t, y_shunt
	# Create the four entries of a Y-Bus matrix for each line.
	yjj = y + 1j*b_line/2
	yii = yjj/(abs(t)**2)
	yij = -y/np.conj(t)
	yji = -y/t
	for k in range(nl):
		i = frombus - 1
		j = tobus - 1
		y_bus[i, j] = yij[k]
		y_bus[j, i] = yji[k]
		y_bus[i, i] += yii[k]
		y_bus[j, j] += yjj[k]
	for i in range(n):
		y_bus[i, i] += y_shunt[i]

	return y_bus
