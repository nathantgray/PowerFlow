from scipy.optimize import fsolve
from math import pi
from numpy import sin, cos
import numpy as np
from power_system import PowerSystem
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mplt




def dynamic_eqns(x, y, df, ps, bus):
	ibus = bus - 1
	u = df.iloc[:, ibus]
	ws = u[0]
	H = u[1]
	Kd = u[2]
	Td0 = u[3]
	Tq0 = u[4]
	xd = u[5]
	xdp = u[6]
	xq = u[7]
	xqp = u[8]
	Rs = u[9]

	th = x[0]
	w = x[1]
	Eqp = x[2]
	Edp = x[3]
	Iq = x[4]
	Id = x[5]
	Pm = x[6]
	Efd = x[7]


	d = y[0][ibus]
	v = y[1][ibus]

	s = ps.complex_injections(y[1], y[0])
	Pg = s.real[ibus]
	Qg = s.imag[ibus]

	vd = v*sin(th - d)
	vq = v*cos(th - d)
	th_dot = (w - 1)*ws
	w_dot = 1/(2*H)*(Pm - Pg - Kd*(w-1))
	Eqp_dot = 1/Td0*(-Eqp - (xd - xdp)*Id + Efd)
	Edp_dot = 1/Tq0*(-Edp + (xq - xqp)*Iq)
	dEqp = vq + Rs*Iq + xdp*Id - Eqp
	dEdp = vd + Rs*Id - xqp*Iq - Edp
	dPg = vd * Id + vq * Iq - Pg
	dQg = vq * Id - vd * Iq - Qg
	return [th_dot, w_dot, Eqp_dot, Edp_dot, dEqp, dEdp, dPg, dQg]

def dyn_jac(x, y, df, ps, bus):
	ibus = bus - 1
	u = df.iloc[:, ibus]
	ws = u[0]
	H = u[1]
	Kd = u[2]
	Td0 = u[3]
	Tq0 = u[4]
	xd = u[5]
	xdp = u[6]
	xq = u[7]
	xqp = u[8]
	Rs = u[9]

	th = x[0]
	Iq = x[4]
	Id = x[5]

	d = y[0][ibus]
	v = y[1][ibus]

	jac = np.zeros((8, len(x)))

	jac[0, 1] = ws

	jac[1, 1] = -Kd/(2*H)
	jac[1, 6] = 1/(2*H)

	jac[2, 2] = -1/Td0
	jac[2, 5] = -1/Td0*(xd - xdp)
	jac[2, 7] = 1/Td0

	jac[3, 3] = -1/Tq0
	jac[3, 4] = 1/Tq0*(xq - xqp)

	jac[4, 0] = -v*sin(th - d)
	jac[4, 2] = -1
	jac[4, 4] = Rs
	jac[4, 5] = xdp

	jac[5, 0] = v*cos(th - d)
	jac[5, 3] = -1
	jac[5, 5] = Rs
	jac[5, 4] = xqp

	jac[6, 0] = v*cos(th - d)*Id - v*sin(th - d)*Iq
	jac[6, 5] = v*sin(th - d)
	jac[6, 4] = v*cos(th - d)

	jac[7, 0] = -v*sin(th - d)*Id - v*cos(th - d)*Iq
	jac[7, 5] = v*cos(th - d)
	jac[7, 4] = -v*sin(th - d)
	return jac

if __name__ == "__main__":
	# case_name = "2BUS.txt"
	case_name = 'Kundur_modified.txt'
	ps = PowerSystem(case_name, sparse=False)
	v0, d0 = ps.flat_start()
	v_nr, d_nr, it = ps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False)
	# y0 = np.r_[d0[1:], v0[1:]]
	# y_s = ps.nr(ps.g, y0, ps.dgdx)
	# ye = np.r_[d_nr, v_nr]
	# jac = ps.dgdx(ye)
	zbase_conv = 0.0008401596
	sbase_conv = 9
	ws = 2 * pi * 60
	H = 6.5 * sbase_conv
	Kd = 0
	Td0 = 8
	Tq0 = 0.4
	xd = 1.8 * zbase_conv
	xdp = 0.3 * zbase_conv
	xq = 1.7 * zbase_conv
	xqp = 0.55 * zbase_conv
	Rs = 0.0025 * zbase_conv


	H1 = 6.5 * sbase_conv
	H3 = 6.175 * sbase_conv
	d = {
		'G1':[ws, H1, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs],
		'G2':[ws, H1, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs],
		'G3':[ws, H3, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs],
		'G4':[ws, H3, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs]
	}
	kundar_dyn = pd.DataFrame(d)
	# u = [ws, H, Kd, Td0, Tq0, xd, xdp, xq, xqp, Rs]
	y = [d_nr, v_nr]
	print(y)
	th0 = d_nr[1]
	w0 = 1
	Eqp0 = v_nr[1]
	Edp0 = 0
	Iq0 = 0.5
	Id0 = 0
	Pm0 = 0.5
	Efd0 = v_nr[1]
	x0 = [th0, w0, Eqp0, Edp0, Iq0, Id0, Pm0, Efd0]
	bus = 2
	x = fsolve(dynamic_eqns, x0, args=(y, kundar_dyn, ps, bus), fprime=dyn_jac)
	th = x[0]
	w = x[1]
	Eqp = x[2]
	Edp = x[3]
	Iq = x[4]
	Id = x[5]
	d = y[0][1]
	vq = -(Rs * Iq + xdp * Id - Eqp)
	vd = -(Rs * Id - xqp * Iq - Edp)
	v = (vd**2 + vq**2)**(1/2)
	print(
		'\nth = ', x[0]*180/pi,
		'\nw = ', x[1],
		'\nEqp = ', x[2],
		'\nEdp = ', x[3],
		'\nIq = ', x[4],
		'\nId = ', x[5],
		'\nPm = ', x[6],
		'\nEfd = ', x[7],
		'\nv = ', v

	)

	fig = plt.figure()
	ax = fig.add_subplot(111, polar=True)
	ax.set_thetamin(0)
	ax.set_thetamax(150)

	angle = pi/2 - th
	print('ref angle: ', angle*180/pi)
	plt.polar([angle, angle], [0, 1], label="ref")

	angle = pi/2 - (th - d_nr[bus - 1])
	print('v angle: ', angle*180/pi)
	plt.polar([angle, angle], [0, v], label="v")
	angle = np.angle(Edp + 1j*Eqp)
	print('E angle: ', angle*180/pi)
	mag = np.abs(Edp + 1j*Eqp)
	plt.polar([angle, angle], [0, mag], label="E")
	# plt.polar([pi/2, pi/2], [0, vq], label="vq")
	# plt.polar([0, 0], [0, vd], label="vd")
	# plt.polar([pi/2, pi/2], [0, Eqp], label="Eqp")
	# plt.polar([0, 0], [0, Edp], label="Edp")
	ax.legend()

	plt.show()