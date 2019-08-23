import numpy as np

def readCase(filename):
	# count rows of bus data
	i = 0
	numBusCol = 18
	with open(filename) as f:
		for line in f:
			# Bus data
			if i >= 2:
				if line[0] == '-':
					numBus = i-2
					break
			i = i + 1
	# Build bus data array
	busData = np.zeros((numBus, numBusCol))
	i = 0
	j = 0
	with open(filename) as f:
		for line in f:
			if i >= 2 and j < numBus:
				if line[0] == '-':
					break
				busData[j, 0] = int(line[1:4])
				busData[j, 1] = int(line[1:4])
				busData[j, 2] = int(line[19:20])
				busData[j, 3] = int(line[21:23])
				busData[j, 4] = int(line[25:26])
				busData[j, 5] = float(line[27:33])
				busData[j, 6] = float(line[34:40])
				busData[j, 7] = float(line[41:49])
				busData[j, 8] = float(line[50:59])
				busData[j, 9] = float(line[60:67])
				busData[j, 10] = float(line[68:75])
				busData[j, 11] = float(line[77:83])
				busData[j, 12] = float(line[85:90])
				busData[j, 13] = float(line[91:98])
				busData[j, 14] = float(line[99:106])
				busData[j, 15] = float(line[107:114])
				busData[j, 16] = float(line[115:122])
				busData[j, 17] = int(line[124:127])

				j = j + 1
			i = i + 1


	branchDataStart = numBus + 4
	i = 0
	numBranchCol = 21
	with open(filename) as f:
		for line in f:
			# Bus data
			if i >= branchDataStart:
				if line[0] == '-':
					numBranch = i-branchDataStart
					break
			i = i + 1
	branchData = np.zeros((numBranch, numBranchCol))
	i = 0
	j = 0
	with open(filename) as f:
		for line in f:
			if i >= branchDataStart and j < numBranch:
				if line[0] == '-':
					break
				branchData[j, 0] = int(line[1:4])
				branchData[j, 1] = int(line[6:9])
				branchData[j, 2] = int(line[11:12])
				branchData[j, 3] = int(line[13:15])
				branchData[j, 4] = int(line[16:17])
				branchData[j, 5] = int(line[18:19])
				branchData[j, 6] = float(line[20:29])
				branchData[j, 7] = float(line[30:40])
				branchData[j, 8] = float(line[41:50])
				branchData[j, 9] = int(line[51:55])
				branchData[j, 10] = int(line[57:61])
				branchData[j, 11] = int(line[63:67])
				branchData[j, 12] = int(line[69:72])
				branchData[j, 13] = int(line[73:74])
				branchData[j, 14] = float(line[76:82])
				branchData[j, 15] = float(line[84:90])
				branchData[j, 16] = float(line[90:97])
				branchData[j, 17] = float(line[97:104])
				branchData[j, 18] = float(line[105:111])
				branchData[j, 19] = float(line[112:118])
				branchData[j, 20] = float(line[118:126])

				j = j + 1
			i = i + 1
	return busData, branchData

if __name__ == "__main__":


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



	filename = 'IEEE14BUS.txt'
	busData, branchData = readCase(filename)
	print(busData)
	print(branchData)
