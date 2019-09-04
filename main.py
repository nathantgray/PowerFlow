import numpy as np
from mismatch import mismatch
from read_testcase import read_case
from makeYbus import makeybus
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

branchFromBus = 0
branchToBus = 1
branchR = 6
branchX = 7
branchB = 8
branchTurnsRatio = 14
branchPhaseShift = 15

filename = 'IEEE14BUS_handout.txt'

# Load case data
busData, branchData, p_base = read_case(filename)

# Get bus types
types = busData[:, busType]
slack = np.where(types == 3)
pv = np.where(types == 2)[0]  # list of PV bus indices
pq = np.where(types < 2)[0]  # list of PQ bus indices
pvpq = np.sort(np.concatenate((pv, pq)))  # list of indices of non-slack buses

# Calculate scheduled P and Q for each bus
mw_gen = busData[pvpq, busGenMW]
mw_load = busData[pvpq, busLoadMW]
mvar_load = busData[pq, busLoadMVAR]
psched = np.array([mw_gen - mw_load]).transpose()/p_base
qsched = np.array([- mvar_load]).transpose()/p_base

# Make the Y-bus matrix
y = makeybus(busData, branchData)

# Initialize with flat start
v = np.array([np.where(busData[:, busDesiredVolts] == 0.0, 1, busData[:, busDesiredVolts])]).transpose()
d = np.zeros_like(v)

# Perform the Newton-Raphson method
v, d, it = pf_newtonraphson(v, d, y, pq, pvpq, psched, qsched, prec=2, maxit=5)

mis, pcalc, qcalc = mismatch(v, d, y, pq, pvpq, psched, qsched)
print("Real Y_Bus: \n", y.real)
print("Imaginary Y_Bus: \n", y.imag)
print("Bus voltages: \n", v)
print("Bus angles (deg): \n", d*180/np.pi)
print("Iterations: ", it)
print("Mismatch: \n", mis)
print("Calculated MW: \n", pcalc*p_base)
print("Calculated MVAR: \n", qcalc*p_base)
