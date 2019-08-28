import numpy as np
from mismatch import mismatch
from read_testcase import readCase
from makeYbus import makeybus
# from PF_Jacobian import pf_jacobian
from PF_NewtonRaphson import pf_newtonraphson

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

filename = 'IEEE14BUS_handout.txt'
busData, branchData = readCase(filename)
types = busData[:, busType]
slack = np.where(types == 3)
pv = np.where(types == 2)[0]  # list of PV bus indices
pq = np.where(types < 2)[0]  # list of PQ bus indices
pvpq = np.sort(np.concatenate((pv, pq)))
psched = np.array([busData[pvpq, busGenMW] - busData[pvpq, busLoadMW]]).transpose()
qsched = np.array([- busData[pq, busLoadMVAR]]).transpose()
y = makeybus(filename)
v = np.array([np.where(busData[:, busDesiredVolts] == 0.0, 1, busData[:, busDesiredVolts])]).transpose()
d = np.zeros_like(v)


# mis, pcalc, qcalc = mismatch(v, d, y, pq, pv, psched, qsched)
# print(mis)

# j, j11, j21, j12, j22= pf_jacobian(v, d, y, pq)
# print(j)
v, d, it = pf_newtonraphson(v, d, y, pq, pv, psched, qsched, prec=2, maxit=5)
mis, pcalc, qcalc = mismatch(v, d, y, pq, pv, psched, qsched)
print(v)
print(d)
print(it)
print(mis)
print(pcalc)
print(qcalc)
