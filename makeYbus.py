from read_testcase import readCase
import numpy as np


def makeybus(filename):
	busShuntG = 15
	busShuntB = 16

	branchR = 6
	branchX = 7
	branchB = 8
	branchTurnsRatio = 14
	branchPhaseShift = 15

	busData, branchData = readCase(filename)

	nl = np.shape(branchData)[0]  # number of lines
	n = np.shape(busData)[0]  # number of buses
	Ybus = np.zeros((n, n)) + np.zeros((n, n))*1j  # initialize Y Bus Matrix
	branchZ = branchData[:, branchR] + branchData[:, branchX]*1j
	branchY = branchZ**-1
	branchB = branchData[:, branchB]
	ratio = np.where(branchData[:, branchTurnsRatio] == 0.0, 1, branchData[:, branchTurnsRatio])
	shift = np.radians(branchData[:, branchPhaseShift])
	tap = ratio*np.cos(shift) + 1j*ratio*np.sin(shift)

	# Create the four entries of a Y-Bus matrix for each line.
	Ytt = branchY + 1j*branchB/2
	Yff = Ytt/(abs(tap)**2)
	Yft = -branchY/np.conj(tap)
	Ytf = -branchY/tap
	# Shunt admittances for each bus.
	Ysh = busData[:, busShuntG] + 1j*busData[:, busShuntB]
	for i in range(nl):
		frombus = int(branchData[i, 0])-1
		tobus = int(branchData[i, 1])-1
		Ybus[frombus, tobus] += Yft[i]
		Ybus[tobus, frombus] += Ytf[i]
		Ybus[frombus, frombus] += Yff[i]
		Ybus[tobus, tobus] += Ytt[i]
	for i in range(n):
		Ybus[i, i] += Ysh[i]

	return Ybus
