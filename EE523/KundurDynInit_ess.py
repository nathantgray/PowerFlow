import numpy as np
from numpy import pi
case_name = 'Kundur_modified.txt'
zbase_conv = 1 / 9
sbase_conv = 9
ws = 2 * pi * 60
H = 6.5 * sbase_conv
Kd = 2 * sbase_conv
Td0 = 8
Tq0 = 0.4
xd = 1.8 * zbase_conv
xdp = 0.3 * zbase_conv
xq = 1.7 * zbase_conv
xqp = 0.55 * zbase_conv
Rs = 0  # 0.0025 * zbase_conv
H1 = 6.5 * sbase_conv
H3 = 6.175 * sbase_conv
vr_min = -4
vr_max = 4
efd_min = 0
efd_max = 2
psg_min = 0 * sbase_conv
psg_max = 1 * sbase_conv
R = 0.05 / sbase_conv  # droop 5%
# Washout filter
tw = 10
# lead compensator
k1 = 5  # 40
wmax = 20  # omega where phase angle is maximum
wmag = 11 * pi / 2  # maximum phase shift
b = 1 / wmag ** 2
t1 = 1 / (wmax * np.sqrt(b))  # 0.05
t2 = t1 * b  # 0.02

udict = {
	'ws': np.array([ws, ws, ws]),
	'H': np.array([H1, H3, H3]),
	'Kd': np.array([Kd, Kd, Kd]),
	'Td0': np.array([Td0, Td0, Td0]),
	'Tq0': np.array([Tq0, Tq0, Tq0]),
	'xd': np.array([xd, xd, xd]),
	'xdp': np.array([xdp, xdp, xdp, xdp]),
	'xq': np.array([xq, xq, xq]),
	'xqp': np.array([xqp, xqp, xqp, xqp]),
	'Rs': np.array([Rs, Rs, Rs, 0.0025 * zbase_conv]),
	'Ka': np.array([50, 50, 50]),
	'Ta': np.array([0.01, 0.01, 0.01]),
	'Vr_min': np.array([vr_min, vr_min, vr_min]),
	'Vr_max': np.array([vr_max, vr_max, vr_max]),
	'Efd_min': np.array([efd_min, efd_min, efd_min]),
	'Efd_max': np.array([efd_max, efd_max, efd_max]),
	'Tsg': np.array([100, 100, 100]),
	'Ksg': np.array([1, 1, 1]),
	'Psg_min': np.array([psg_min, psg_min, psg_min]),
	'Psg_max': np.array([psg_max, psg_max, psg_max]),
	'R': np.array([R, R, R]),
	'Tw': np.array([tw, tw, tw]),  # Typically between 1 and 20 seconds. Kundur pg 1133
	'K1': np.array([k1, k1, k1]),
	'T1': np.array([t1, t1, t1]),
	'T2': np.array([t2, t2, t2]),
	'comp': np.array([1, 1, 1]),
	'constZ': 0.5,
	'constP': 0.5,
	'constI': 0
}