from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from classes.power_system import PowerSystem
from copy import deepcopy
"""
 1, 1.03000,   0.0000
 2, 1.01000,  -9.3421
 3, 1.03000, -19.0129
 4, 1.01000, -29.0817
 5, 1.02033,  -6.2952
 6, 1.01184, -15.8986
 7, 1.02133, -23.6496
 8, 1.00955, -31.6612
 9, 1.00254, -43.8019
10, 1.00081, -35.7108
11, 1.01533, -25.5940
"""

# part 0
# Create Ygen
case_name = "KundurEx12-6_hw1b.txt"
# case_name = "KundurEx12-6.txt"
ps = PowerSystem(case_name, sparse=False)
v0, d0 = ps.flat_start()
v_nr, d_nr, it_nr = ps.pf_newtonraphson(v0, d0, prec=3, maxit=8, qlim=False)
ybus = ps.y_bus
S7 = 9.67 + 1j
S9 = 17.67 + 1j
v7 = 1.02133
v9 = 1.00254
yL7 = np.conj(S7)/v7**2
yL9 = np.conj(S9)/v9**2
yc7 = 4j
yc9 = 4j
y7 = yL7 + yc7
y9 = yL9 + yc9
yg = -2801.12j
Y11 = deepcopy(ybus)
Y12 = np.zeros((11, 3)) + np.zeros((11, 3))*1j
Y21 = np.zeros((3, 11)) + np.zeros((3, 11))*1j
Y22 = np.zeros((3, 3)) + np.zeros((3, 3))*1j

Y12[5, 0] = ybus[5, 1]
Y11[5, 1] = 0
Y21[0, 5] = ybus[1, 5]
Y11[1, 5] = 0
Y12[9, 2] = ybus[9, 3]
Y11[9, 3] = 0
Y21[2, 9] = ybus[3, 9]
Y11[3, 9] = 0
Y12[10, 1] = ybus[10, 2]
Y11[10, 2] = 0
Y21[1, 10] = ybus[2, 10]
Y11[2, 10] = 0
Y12[1, 0] = -yg
Y12[2, 1] = -yg
Y12[3, 2] = -yg
Y21[0, 1] = -yg
Y21[1, 2] = -yg
Y21[2, 3] = -yg
Y11[1, 1] = Y11[1, 1] + yg
Y11[2, 2] = Y11[2, 2] + yg
Y11[3, 3] = Y11[3, 3] + yg
Y11[6, 6] = Y11[6, 6] + yL7
Y11[8, 8] = Y11[8, 8] + yL9
Y22[0, 0] = sum([-y for y in Y12[:, 0]])
Y22[1, 1] = sum([-y for y in Y12[:, 1]])
Y22[2, 2] = sum([-y for y in Y12[:, 2]])
Ygen = Y11 - Y12 @ inv(Y22) @ Y21
print(yL7)
print('stop')

# Part 1
s = tf('s')
G = 10/(1+s*2)
dt = 0.01
T = np.arange(0, 30, dt) - 5
U = np.zeros(len(T))
U[int(5/dt):int(10/dt)] = 1
U[int(10/dt):int(15/dt)] = -1
# Ua = np.where(U < 5, U, 5)
# Ua = np.where(Ua > -5, Ua, -5)
y1, T, xout = lsim(G, U, T)
y1a = np.where(y1 < 5, y1, 5)
y1a = np.where(y1a > -5, y1a, -5)
V = np.zeros(len(T))
V[int(5/dt):int(10/dt)] = np.arange(0, 1, 1/500)
V[int(10/dt):int(20/dt)] = np.arange(1, -1, -1/500)
V[int(20/dt):int(25/dt)] = np.arange(-1, -0, 1/500)
y2, T, xout = lsim(G, V, T)
y2a = np.where(y2 < 5, y2, 5)
y2a = np.where(y2a > -5, y2a, -5)


y1b = np.zeros(len(T))
for n in range(len(T)-1):
	y1b[n + 1] = 5 * U[n] * dt - 1 / 2 * y1b[n] * dt + y1b[n]
	ydot = (y1b[n + 1] - y1b[n]) / dt
	if (y1b[n + 1] >= 5 and (ydot > 0)) or (y1b[n + 1] <= -5 and (ydot < 0)):
		y1b[n + 1] = y1b[n]

y2b = np.zeros(len(T))
for n in range(len(T)-1):
	y2b[n + 1] = 5 * V[n] * dt - 1 / 2 * y2b[n] * dt + y2b[n]
	ydot = (y2b[n + 1] - y2b[n]) / dt
	if (y2b[n + 1] >= 5 and (ydot > 0)) or (y2b[n + 1] <= -5 and (ydot < 0)):
		y2b[n + 1] = y2b[n]

plt.figure(1)
plt.subplot(211)
plt.plot(T, U*10, label='10*u (a)')
plt.plot(T, y1, label='y (no limit)')
plt.plot(T, y1a, label='y (windup)')
plt.plot(T, y1b, label='y (non-windup)')
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper right', ncol=2, borderaxespad=0.)
plt.title('Response to Input Signal a')
plt.subplot(212)
plt.plot(T, V*10, label='10*u (b)')
plt.plot(T, y2, label='y (no limit)')
plt.plot(T, y2a, label='y (windup)')
plt.plot(T, y2b, label='y (non-windup)')
plt.legend(bbox_to_anchor=(1.0, 1), loc='upper right', ncol=2, borderaxespad=0.)
plt.title('Response to Input Signal b')
plt.show()



