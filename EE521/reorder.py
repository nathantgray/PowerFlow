import numpy as np


def reorder(busData, branchData):

	busNumber = 0
	busType = 4
	types = busData[:, busType]
	oldBusNumbers = busData[:, busNumber]
	newBusIndex = np.concatenate(np.where(types == 3) + np.where(types == 2) + np.where(types < 2))
	newBusNumbers = newBusIndex + np.ones_like(newBusIndex)
	newBusData = busData[newBusIndex, :]
	newBusData[:, 0] = oldBusNumbers
	for i in range(np.shape(branchData[:, 0])[0]):
		branchData[i, 0] = newBusNumbers[int(branchData[i, 0]) - 1]
		branchData[i, 1] = newBusNumbers[int(branchData[i, 1]) - 1]
	busData = newBusData

	np.set_printoptions(linewidth=np.inf)
	print(newBusData)

	return busData, branchData
