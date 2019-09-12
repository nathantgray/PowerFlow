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
				branch_data[j, 0] = int(line[0:4])             # Columns  1- 4   Tap bus number (I) *
				branch_data[j, 1] = int(line[5:9])             # Columns  6- 9   Z bus number (I) *
				branch_data[j, 2] = int(line[10:12])           # Columns 11-12   Load flow area (I)
				branch_data[j, 3] = int(line[12:15])           # Columns 13-14   Loss zone (I)
				branch_data[j, 4] = int(line[16:17])           # Column  17      Circuit (I) * (Use 1 for single lines)
				branch_data[j, 5] = int(line[18:19])           # Column  19      Type (I) *
				branch_data[j, 6] = float(line[19:29])         # Columns 20-29   Branch resistance R, per unit (F) *
				branch_data[j, 7] = float(line[29:40])         # Columns 30-40   Branch reactance X, per unit (F) *
				branch_data[j, 8] = float(line[40:50])         # Columns 41-50   Line charging B, per unit (F) *
				branch_data[j, 9] = int(line[50:55])           # Columns 51-55   Line MVA rating No 1 (I) Left justify!
				branch_data[j, 10] = int(line[56:61])          # Columns 57-61   Line MVA rating No 2 (I) Left justify!
				branch_data[j, 11] = int(line[62:67])          # Columns 63-67   Line MVA rating No 3 (I) Left justify!
				branch_data[j, 12] = int(line[68:72])          # Columns 69-72   Control bus number
				branch_data[j, 13] = int(line[73:74])          # Column  74      Side (I)
				branch_data[j, 14] = float(line[75:82])        # Columns 77-82   Transformer final turns ratio (F)
				branch_data[j, 15] = float(line[83:90])        # Columns 84-90   Transformer (phase shifter) final angle (F)
				branch_data[j, 16] = float(line[90:97])        # Columns 91-97   Minimum tap or phase shift (F)
				branch_data[j, 17] = float(line[97:104])       # Columns 98-104  Maximum tap or phase shift (F)
				branch_data[j, 18] = float(line[105:111])      # Columns 106-111 Step size (F)
				branch_data[j, 19] = float(line[112:118])      # Columns 113-119 Minimum voltage, MVAR or MW limit (F)
				branch_data[j, 20] = float(line[119:126])      # Columns 120-126 Maximum voltage, MVAR or MW limit (F)

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
