from read_testcase import readCase
from reorder import reorder
import numpy as np
from math import cos, sin, pi, radians
def makeYbus(filename):
	busNumber = 0
	busArea = 2
	busZone = 3
	busType = 4
	busFinalVoltage = 5
	busFinalAngle = 6
	busLoadMW = 7
	busLoadMVAR = 8
	busGenMW = 9
	busGenMVAR = 10
	busBaseKV = 11
	busDesiredVolts = 12
	busMaxMVAR = 13
	busMinMVAR = 14
	busShuntG = 15
	busShuntB = 16
	busRemoteControlledBusNumber = 17

	branchTapBus = 0
	branchZBus = 1


	branchR = 6
	branchX = 7
	branchB = 8
	branchTurnsRatio = 14
	branchPhaseShift = 15

	busData, branchData = readCase(filename)
	#busData, branchData = reorder(busData, branchData)

	nl = np.shape(branchData)[0]  # number of lines
	n = np.shape(busData)[0]  # number of buses
	Ybus = np.zeros((n, n)) + np.zeros((n, n))*1j # initialize Y Bus Matrix
	branchZ = branchData[:, branchR] + branchData[:, branchX]*1j
	branchY = branchZ**-1
	branchB = branchData[:, branchB]
	ratio = np.where(branchData[:, branchTurnsRatio]==0.0, 1, branchData[:, branchTurnsRatio])
	shift = np.radians(branchData[:, branchPhaseShift])
	tap = ratio*np.cos(shift) + 1j*ratio*np.sin(shift)

	# Create the four entries of a Y-Bus matrix for each line.
	Ytt = branchY + 1j*branchB/2
	Yff = Ytt/(abs(tap)**2)
	Yft = -Ytt/np.conj(tap)
	Ytf = -Ytt/tap
	# Shunt admittances for each bus.
	Ysh =  busData[:, busShuntG]+ 1j*busData[:, busShuntB]
	for i in range(nl):
		frombus = int(branchData[i, 0])-1
		tobus = int(branchData[i,1])-1
		Ybus[frombus, tobus] += Yft[i]
		Ybus[tobus, frombus] += Ytf[i]
		Ybus[frombus, frombus] += Yff[i]
		Ybus[tobus, tobus] += Ytt[i]
	for i in range(n):
		Ybus[i,i] += Ysh[i]

	#print(Ybus)
	#print('\n')
	#print(branchData)
	return Ybus