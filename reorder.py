import numpy as np

def reorder(busData, branchData):
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
	types = busData[:, busType]
	i = 0
	oldBusNumbers = busData[:, busNumber]
	newBusIndex = np.concatenate(np.where(types==3) + np.where(types==2) + np.where(types<2))
	newBusNumbers = newBusIndex + np.ones_like(newBusIndex)
	newBusData = busData[newBusIndex, :]
	newBusData[:, 0] = oldBusNumbers
	for i in range(np.shape(branchData[:,0])[0]):
		branchData[i, 0] = newBusNumbers[int(branchData[i,0])-1]
		branchData[i, 1] = newBusNumbers[int(branchData[i,1])-1]
	busData = newBusData



	np.set_printoptions(linewidth=np.inf)
	print(newBusData)

	return busData, branchData