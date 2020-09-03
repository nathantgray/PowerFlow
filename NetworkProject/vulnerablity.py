from classes.power_system import *
from numpy.linalg import inv
import copy
import matplotlib.pyplot as plt
ps = PowerSystem('case1_heavy.txt', sparse=False)
# v0 = np.array([1, 1, 1, 1, 0.99998393, 0.99998191, 0.99999415, 1, 1, 1, 1, 1., 1, 1, 0.99830667, 0.98453577, 0.97525913, 0.96744577, 0.99403661, 0.99426043, 0.99493397, 0.99644641, 0.99811352, 0.99905194, 0.94998487, 0.95180199, 0.95181769, 0.95981856, 0.95981856, 0.99998006, 0.99406179, 0.98914975, 0.98681977, 0.98914415, 0.98619281, 0.9857869, 0.98570378, 0.98568945, 0.98773372, 0.99343952, 0.99351437, 0.99360284, 0.99371767, 0.99544489, 0.9972126, 0.99829061, 0.99938774, 0.99999864, 1, 1, 1, 1, 1, 0.99999346, 0.99997225])
# d0 = np.array([0.00000000e+00,  1.84434164e-01,  1.84462092e-01,  1.84405938e-01,  1.84191207e-01,  1.84138680e-01, 9.33957289e-02,  9.34171772e-02,  9.34151108e-02,  9.34134979e-02,  9.34016876e-02,  9.34318769e-02,  9.33898773e-02,  9.34016876e-02,  9.23078802e-02,  8.32818107e-02,  7.70740645e-02,  7.17424600e-02, -3.97296145e-03, -3.82259891e-03, -3.37052713e-03, -2.36025588e-03, -1.25068836e-03, -6.27834897e-04, -3.31589526e-02, -3.18808576e-02, -3.18644115e-02, -2.63372278e-02, -2.63372278e-02, -1.97282834e-05, -3.93102150e-03, -7.21587294e-03, -8.78870146e-03, -7.22152414e-03, -9.20827319e-03, -9.48274203e-03, -9.53896961e-03, -9.54866498e-03, -8.17183505e-03, -4.36976728e-03, -4.31943680e-03, -4.25996511e-03, -4.18278207e-03, -3.02678864e-03, -1.84868988e-03, -1.13277530e-03, -4.06006079e-04,  -1.34647885e-06, 1.84405938e-01, 0.00000000e+00, 6.25291419e-01, 6.25322919e-01, 6.25291419e-01, 6.25234959e-01, 4.96062707e-05])
v = np.genfromtxt('case0_heavy_v0.csv')
d = np.genfromtxt('case0_heavy_d0.csv')
p_loss = np.array([])
# v0, d0 = ps.pf(initial=(v0, d0))
gens = np.r_[ps.slack, ps.pv]
loads = np.where(ps.p_load_full > 0)[0]
y_bus = copy.deepcopy(ps.y_bus)
z_bus = inv(y_bus)
vulnerability = np.zeros_like(v)
NA0 = 0
for i in gens:
	for j in loads:
		z = z_bus[i, i] + z_bus[j, j] - z_bus[i, j] - z_bus[j, i]
		NA0 = NA0 + ps.p_gen_full[i] / (ps.p_load_full[j] * np.exp(np.abs(z)))

k = 0
for fbus in range(len(v)):
	y_busf = copy.deepcopy(y_bus)
	for tbus in range(len(v)):
		if tbus != fbus:
			y_busf[fbus, tbus] = 0
			y_busf[tbus, fbus] = 0
	z_bus = inv(y_busf)
	NA = 0
	for i in gens:
		for j in loads:
			z = z_bus[i, i] + z_bus[j, j] - z_bus[i, j] - z_bus[j, i]
			if z_bus[i, j] < 1e-10:
				z = np.inf
			NA = NA + ps.p_gen_full[i] / (ps.p_load_full[j] * np.exp(np.abs(z)))
	vulnerability[k] = (NA0 - NA)/NA0
	k = k + 1
print(vulnerability)
plt.plot(vulnerability)
# y_busf = copy.deepcopy(y_bus)
# y_busf[6, 0] = 0
# y_busf[0, 6] = 0
# z_bus = inv(y_busf)
# for i in gens:
# 	for j in loads:
# 		z = z_bus[i, i] + z_bus[j, j] - z_bus[i, j] - z_bus[j, i]
# 		if z_bus[i, j] < 1e-10:
# 			z = np.inf
# 		NA = NA + ps.p_gen_full[i] / (ps.p_load_full[j] * np.exp(np.abs(z)))
# print(NA)
# vulnerability = (NA0 - NA)/NA0
# print(vulnerability)
