import numpy as np
from mismatch import mismatch
from read_testcase import readCase
from makeYbus import makeYbus
from PF_Jacobian import PF_Jacobian
from reorder import reorder

np.set_printoptions(linewidth=np.inf, precision=4, suppress=True)


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

filename = 'IEEE14BUS.txt'
busData, branchData = readCase(filename)
types = busData[:, busType]
slack = np.where(types == 3)
pv = np.where(types == 2)[0]
pq = np.where(types < 2)[0]
psched = busData[np.concatenate((pv,pq)), busGenMW] - busData[np.concatenate((pv,pq)), busLoadMW]
qsched = busData[pq, busGenMVAR] - busData[pq, busLoadMVAR]
y = makeYbus(filename)
v = np.where(busData[:, busDesiredVolts] == 0.0, 1, busData[:, busDesiredVolts])
d = np.zeros_like(v)
mis, pcalc, qcalc = mismatch(v, d, y, pq, pv, psched, qsched)
print(mis)

j, j11, j21, j12, j22= PF_Jacobian(v, d, y, pq)

print(y)
print(j)
print(j11)
print(j12)
print(j21)
print(j22)