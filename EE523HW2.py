import pandas as pd
from power_system import PowerSystem
import numpy as np
import sympy as sm
from sympy.plotting import plot, PlotGrid
from sympy.integrals import laplace_transform, inverse_laplace_transform
from sympy.functions.special.delta_functions import Heaviside
import numpy as np

e = np.e
t = sm.symbols('t')
s = sm.symbols('s')
u10 = Heaviside(t - 10)
u20 = Heaviside(t - 20)
v = 1 - 0.1 * u10 + 0.05 * u20
pd = -0.05 * u10 + 0.05 * e ** -((t - 10) / 5) * u10 + 0.025 * u20 - 0.025 * e ** -((t - 20) / 5) * u20
qd = -0.1 * u10 * (0.2 - 0.2 * e ** -((t - 10) / 3)) + 0.05 * u20 * (0.2 - 0.2 * e ** -((t - 20) / 3))
qs = 0.8 * v ** 0.65
ps = 0.5 * v ** 0.75


# pt0 = pt.subs(t, 0)
# pt10 = pt.subs(t, 10.000001)
# pt15 = pt.subs(t, 15)
# pt19 = pt.subs(t, 19.999999)
# pt20 = pt.subs(t, 20.000001)
# pt25 = pt.subs(t, 25)
# p1 = plot(qt, (t, 0, 30), show=True)
def P_sr_a(t):
	return 0.5 - 0.5 * e ** -(t / 5)


def Q_sr_a(t):
	return 0.2 - 0.2 * e ** -(t / 3)


pd_a = -0.1 * u10 * P_sr_a(t - 10) + 0.05 * u20 * P_sr_a(t - 20)
qd_a = -0.1 * u10 * Q_sr_a(t - 10) + 0.05 * u20 * Q_sr_a(t - 20)

pt_a = (0.5 + ps + pd_a) * 100
qt_a = (0.2 + qs + qd_a) * 50
# p2 = plot(pt_a, (t, 0, 30), show=True)

## Part B
def P_sr_b(t):
	wd = (40 - 0.04 ** 2) ** (1 / 2)
	return 0.02 * (1 - e ** (-0.04 * t) * (sm.cos(wd * t) + 0.04 / wd * sm.sin(wd * t)))


def Q_sr_b(t):
	wd = (20 - 0.04 ** 2) ** (1 / 2)
	return 0.02 * (1 - e ** (-0.02 * t) * (sm.cos(wd * t) + 0.04 / wd * sm.sin(wd * t)))


pd_b = -0.1 * u10 * P_sr_b(t - 10) + 0.05 * u20 * P_sr_b(t - 20)
qd_b = -0.1 * u10 * Q_sr_b(t - 10) + 0.05 * u20 * Q_sr_b(t - 20)
pt_b = (0.5 + ps + pd_b) * 100
qt_b = (0.2 + qs + qd_b) * 50
# p_b_plot = plot(pt_b, (t, 0, 30), show=False, adaptive=False)
# q_b_plot = plot(qt_b, (t, 0, 30), show=False, adaptive=False)
# b_plot = p_b_plot.append(q_b_plot[0])
# b_plot.show()
p1 = plot(pt_a, (t, 0, 40), show=False, adaptive=False)
p2 = plot(qt_a, (t, 0, 40), show=False, adaptive=False)
p3 = plot(pt_b, (t, 0, 40), show=False, adaptive=False)
p4 = plot(qt_b, (t, 0, 40), show=False, adaptive=False)
PlotGrid(4, 1, p1, p2, p3, p4)
## Part C
u30 = Heaviside(t - 30)
v_c = 1 + 0.05 * u10 - 0.1 * u20 + 0.05*u30
qs_c = 0.8 * v_c ** 0.65
ps_c = 0.5 * v_c ** 0.75

pd_ca = 0.05 * u10 * P_sr_a(t - 10) - 0.1 * u20 * P_sr_a(t - 20) + 0.05 * u30 * P_sr_a(t - 30)
qd_ca = 0.05 * u10 * Q_sr_a(t - 10) - 0.1 * u20 * Q_sr_a(t - 20) + 0.05 * u30 * Q_sr_a(t - 30)
pt_ca = (0.5 + ps_c + pd_ca) * 100
qt_ca = (0.2 + qs_c + qd_ca) * 50

pd_cb = 0.05 * u10 * P_sr_b(t - 10) - 0.1 * u20 * P_sr_b(t - 20) + 0.05 * u30 * P_sr_b(t - 30)
qd_cb = 0.05 * u10 * Q_sr_b(t - 10) - 0.1 * u20 * Q_sr_b(t - 20) + 0.05 * u30 * Q_sr_b(t - 30)
pt_cb = (0.5 + ps_c + pd_cb) * 100
qt_cb = (0.2 + qs_c + qd_cb) * 50

p1 = plot(pt_ca, (t, 0, 40), show=False, adaptive=False)
p2 = plot(qt_ca, (t, 0, 40), show=False, adaptive=False)
p3 = plot(pt_cb, (t, 0, 40), show=False, adaptive=False)
p4 = plot(qt_cb, (t, 0, 40), show=False, adaptive=False)
PlotGrid(4, 1, p1, p2, p3, p4)