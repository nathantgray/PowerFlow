from read_testcase import readCase
import numpy as np
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
busShuntConductance = 15
busShuntSusceptance = 16
busRemoteControlledBusNumber = 17

branchTapBus = 0
branchZBus = 1


branchR = 6
branchX = 7
branchB = 8
branchTurnsRatio = 14
branchPhaseShift = 15

filename = 'IEEE14BUS.txt'
busData, branchData = readCase(filename)
n = np.shape(busData)[0]
Ybus = np.zeros((n, n)) + np.zeros((n, n))*1j
branchZ = branchData[:, branchR] + branchData[:, branchX]*1j
branchY = branchZ**-1

for i in range(len(branchData[:, branchTapBus])):
	fbus = int(branchData[i, branchTapBus])-1
	tbus = int(branchData[i, branchZBus]) - 1
	Ybus[tbus, fbus] = -branchY[i]
	Ybus[fbus, tbus] = -branchY[i]
np.set_printoptions(linewidth=np.inf)
print(Ybus)
