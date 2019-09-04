import numpy as np


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
					bus_rows = i-2
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
				bus_data[j, 0] = int(line[1:4])
				bus_data[j, 1] = int(line[1:4])
				bus_data[j, 2] = int(line[19:20])
				bus_data[j, 3] = int(line[21:23])
				bus_data[j, 4] = int(line[25:26])
				bus_data[j, 5] = float(line[27:33])
				bus_data[j, 6] = float(line[34:40])
				bus_data[j, 7] = float(line[41:49])
				bus_data[j, 8] = float(line[50:59])
				bus_data[j, 9] = float(line[60:67])
				bus_data[j, 10] = float(line[68:75])
				bus_data[j, 11] = float(line[77:83])
				bus_data[j, 12] = float(line[84:90])
				bus_data[j, 13] = float(line[91:98])
				bus_data[j, 14] = float(line[99:106])
				bus_data[j, 15] = float(line[107:114])
				bus_data[j, 16] = float(line[115:122])
				bus_data[j, 17] = int(line[124:127])

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
					branch_rows = i-branchDataStart
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
				branch_data[j, 0] = int(line[1:4])
				branch_data[j, 1] = int(line[6:9])
				branch_data[j, 2] = int(line[11:12])
				branch_data[j, 3] = int(line[13:15])
				branch_data[j, 4] = int(line[16:17])
				branch_data[j, 5] = int(line[18:19])
				branch_data[j, 6] = float(line[20:29])
				branch_data[j, 7] = float(line[30:40])
				branch_data[j, 8] = float(line[41:50])
				branch_data[j, 9] = int(line[51:55])
				branch_data[j, 10] = int(line[57:61])
				branch_data[j, 11] = int(line[63:67])
				branch_data[j, 12] = int(line[69:72])
				branch_data[j, 13] = int(line[73:74])
				branch_data[j, 14] = float(line[76:82])
				branch_data[j, 15] = float(line[84:90])
				branch_data[j, 16] = float(line[90:97])
				branch_data[j, 17] = float(line[97:104])
				branch_data[j, 18] = float(line[105:111])
				branch_data[j, 19] = float(line[112:118])
				branch_data[j, 20] = float(line[118:126])

				j = j + 1
			i = i + 1
	return bus_data, branch_data, mva_base


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
	busData, branchData, p_base = read_case(filename)
	print(busData)
	print(branchData)
