from classes.dynamics import *
import control as ctrl
import pandas as pd
from EE523.KundurDynInit import *
import matplotlib.pyplot as plt
from scipy.linalg import eig
if __name__ == "__main__":
	case_name = 'CaseFiles/Kundur_modified.txt'

	dps = DynamicSystem(case_name, udict, sparse=False)
	v0, d0 = dps.flat_start()
	v, d, it = dps.pf_newtonraphson(v0, d0, prec=7, maxit=10, qlim=False, verbose=False)
	y = np.r_[d[1:], v[1:]]
	type1 = dps.type1_init()
	type2 = dps.type2_init()
	type3 = dps.type3_init()
	# Check type 1
	th1 = type1['th']
	w1 = np.ones(len(dps.pv))
	Eqp1 = type1['Eqp']
	Edp1 = type1['Edp']
	va1 = type1['Efd']
	Pm1 = type1['Pg']
	x1 = np.array([])
	for i in range(len(dps.pv)):
		x1 = np.r_[x1, th1[i], w1[i], Eqp1[i], Edp1[i], va1[i], Pm1[i]]
	x_dot = dps.dyn1_f(x1, y, type1['vref'], type1['Pc'])
	print('Type 1 max error', np.max(np.abs(x_dot)))
	# Check type 2
	th2 = type2['th']
	w2 = np.ones(len(dps.pv))
	Eqp2 = type2['Eqp']
	Edp2 = type2['Edp']
	va2 = type2['Efd']
	Pm2 = type2['Pg']
	x2 = np.array([])
	for i in range(len(dps.pv)):
		x2 = np.r_[x2, th2[i], w2[i], Eqp2[i], Edp2[i], va2[i], Pm2[i]]
	x_dot = dps.dyn2_f(x2, type2['vref'], type2['Pc'])
	print('Type 2 max error', np.max(np.abs(x_dot)))
	# Check type 3
	th3 = type3['th']
	w3 = np.ones(len(dps.pv))
	Pm3 = type3['Pg']
	E3 = type3['E']
	x3 = np.array([])
	for i in range(len(dps.pv)):
		x3 = np.r_[x3, th3[i], w3[i]]
	x_dot = dps.dyn3_f(x3, E3, Pm3)
	print('Type 3 max error', np.max(np.abs(x_dot)))
	# Check type 1 with compensation
	vw = np.array([0, 0, 0])
	vs = np.array([0, 0, 0])
	x1_comp = np.array([])
	for i in range(len(dps.pv)):
		x1_comp = np.r_[x1_comp, th1[i], w1[i], Eqp1[i], Edp1[i], va1[i], Pm1[i], vw[i], vs[i]]
	x_dot = dps.dyn1_f_comp(x1_comp, y, type1['vref'], type1['Pc'])
	print('Type 1 with compensation max error', np.max(np.abs(x_dot)))
	# Check type 2 with compensation
	x2_comp = np.array([])
	for i in range(len(dps.pv)):
		x2_comp = np.r_[x2_comp, th2[i], w2[i], Eqp2[i], Edp2[i], va2[i], Pm2[i], vw[i], vs[i]]
	x_dot = dps.dyn2_f_comp(x2_comp, type2['vref'], type2['Pc'])
	print('Type 2 with compensation max error', np.max(np.abs(x_dot)))

	J1 = dps.J_dyn1(x1, y, type1['vref'], type1['Pc'])
	ev1, vl1, vr1 = eig(J1, left = True)
	w1 = inv(vr1)
	J2 = dps.J_dyn2(x2, type2['vref'], type2['Pc'])
	ev2, vl2, vr2 = eig(J2, left = True)
	w2 = inv(vr2)
	J3 = dps.J_dyn3(x3, E3, Pm3)
	ev3, vl3, vr3 = eig(J3, left = True)
	w3 = inv(vr3)

	J1_comp = dps.J_dyn1_comp(x1_comp, y, type1['vref'], type1['Pc'])
	ev1_comp, vl1_comp, vr1_comp = eig(J1_comp, left = True)
	w1_comp = inv(vr1_comp)

	J2_comp = dps.J_dyn2_comp(x2_comp, type2['vref'], type2['Pc'])
	ev2_comp, vl2_comp, vr2_comp = eig(J2_comp, left = True)
	w2_comp = inv(vr2_comp)

	f1 = ev1.imag/(2*pi)
	f2 = ev2.imag/(2*pi)
	f3 = ev3.imag/(2*pi)
	f1_comp = ev1_comp.imag / (2 * pi)
	f2_comp = ev2_comp.imag / (2 * pi)
	zeta1 = ev1.real/np.abs(ev1)
	zeta2 = ev2.real/np.abs(ev2)
	zeta3 = ev3.real/np.abs(ev3)
	zeta1_comp = ev1_comp.real/np.abs(ev1_comp)
	zeta2_comp = ev2_comp.real/np.abs(ev2_comp)
	damp1 = zeta1 * -100
	damp2 = zeta2 * -100
	damp3 = zeta3 * -100
	damp1_comp = zeta1_comp * -100
	damp2_comp = zeta2_comp * -100


	threshold = 0.01
	states1 = ['th2', 'w2', 'Eqp2', 'Edp2', 'Efd2', 'Pm2',
			  'th3', 'w3', 'Eqp3', 'Edp3', 'Efd3', 'Pm3',
			  'th4', 'w4', 'Eqp4', 'Edp4', 'Efd4', 'Pm4']
	states2 = ['th2', 'w2', 'Eqp2', 'Edp2', 'Efd2', 'Pm2',
			  'th3', 'w3', 'Eqp3', 'Edp3', 'Efd3', 'Pm3',
			  'th4', 'w4', 'Eqp4', 'Edp4', 'Efd4', 'Pm4']
	states3 = ['th2', 'w2',
			  'th3', 'w3',
			  'th4', 'w4',]
	states_pss = ['th2', 'w2', 'Eqp2', 'Edp2', 'Efd2', 'Pm2', 'vw2', 'vs2',
			  'th3', 'w3', 'Eqp3', 'Edp3', 'Efd3', 'Pm3', 'vw3', 'vs3',
			  'th4', 'w4', 'Eqp4', 'Edp4', 'Efd4', 'Pm4', 'vw4', 'vs4']
	d = {"Eigenvalue": np.round(ev1, 4), "Freq (Hz)": np.round(f1, 3), "Damping %": np.round(damp1, 3)}
	p_f1 = np.zeros_like(J1)
	print('\n-----Type 1-----')
	for k in range(len(x1)):
		print('***Mode:', k, ', Damping:', round(damp1[k], 3), ', Freq:', round(f1[k], 3), ', ', round(ev1[k], 4))
		for i in range(len(x1)):
			p_f1[i, k] = np.abs(vr1[i, k]) * np.abs(w1[k, i]) / np.max(np.abs(vr1[:, k] * w1[k, :]))
			if p_f1[i, k] > threshold and i%6==0:
				print(states1[i], ':', round(p_f1[i, k], 2), sep='', end=' ')
		print('\n')
	d["G2"]=np.round(p_f1[0, :], 2)
	d["G3"]=np.round(p_f1[6, :], 2)
	d["G4"]=np.round(p_f1[12, :], 2)
	df1 = pd.DataFrame(d)
	df1.to_csv('Type1_response.csv')

	print('\n-----Type 2-----')
	d = {"Eigenvalue": np.round(ev2, 4), "Freq (Hz)": np.round(f2, 3), "Damping %": np.round(damp2, 3)}
	p_f2 = np.zeros_like(J2)
	for k in range(len(x2)):
		print('***Mode:', k, ', Damping:', round(damp2[k], 3), ', Freq:', round(f2[k],3), ', ', round(ev2[k], 4))
		for i in range(len(x2)):
			p_f2[i, k] = np.abs(vr2[i, k]) * np.abs(w2[k, i]) / np.max(np.abs(vr2[:, k] * w2[k, :]))
			if p_f2[i, k] > threshold and i%6==0:
				print(states2[i], ':', round(p_f2[i, k], 2), sep='', end=' ')
		print('\n')
	d["G2"]=np.round(p_f2[0, :], 2)
	d["G3"]=np.round(p_f2[6, :], 2)
	d["G4"]=np.round(p_f2[12, :], 2)
	df2 = pd.DataFrame(d)
	df2.to_csv('Type2_response.csv')

	print('\n-----Type 3-----')
	d = {"Eigenvalue": np.round(ev3, 4), "Freq (Hz)": np.round(f3, 3), "Damping %": np.round(damp3, 3)}
	p_f3 = np.zeros_like(J3)
	for k in range(len(x3)):
		print('***Mode:', k, ', Damping:', round(damp3[k], 3), ', Freq:', round(f3[k], 3), ', ', round(ev3[k], 4))
		for i in range(len(x3)):
			p_f3[i, k] = np.abs(vr3[i, k]) * np.abs(w3[k, i]) / np.max(np.abs(vr3[:, k] * w3[k, :]))
			if p_f3[i, k] > threshold:
				print(states3[i], ':', round(p_f3[i, k], 2), sep='', end=' ')
		print('\n')
	d["G2"]=np.round(p_f3[0, :], 2)
	d["G3"]=np.round(p_f3[2, :], 2)
	d["G4"]=np.round(p_f3[4, :], 2)
	df3 = pd.DataFrame(d)
	df3.to_csv('Type3_response.csv')

	p_f1_comp = np.zeros_like(J1_comp)
	print('\n-----Type 1 with Compensation-----')
	d = {"Eigenvalue": np.round(ev1_comp, 4), "Freq (Hz)": np.round(f1_comp, 3), "Damping %": np.round(damp1_comp, 3)}
	for k in range(len(x1_comp)):
		print('***Mode:', k, ', Damping:', round(damp1_comp[k], 3), ', Freq:', round(f1_comp[k], 3), ', ', round(ev1_comp[k], 4))
		for i in range(len(x1_comp)):
			p_f1_comp[i, k] = np.abs(vr1_comp[i, k]) * np.abs(w1_comp[k, i]) / np.max(np.abs(vr1_comp[:, k] * w1_comp[k, :]))
			if p_f1_comp[i, k] > threshold and i%8==0:
				print(states_pss[i], ':', round(p_f1_comp[i, k], 2), sep='', end=' ')
		print('\n')
	d["G2"]=np.round(p_f1_comp[0, :], 2)
	d["G3"]=np.round(p_f1_comp[8, :], 2)
	d["G4"]=np.round(p_f1_comp[16, :], 2)
	df1_comp = pd.DataFrame(d)
	df1_comp.to_csv('Type1_PSS_response.csv')

	p_f2_comp = np.zeros_like(J2_comp)
	print('\n-----Type 2 with Compensation-----')
	d = {"Eigenvalue": np.round(ev2_comp, 4), "Freq (Hz)": np.round(f2_comp, 3), "Damping %": np.round(damp2_comp, 3)}
	for k in range(len(x2_comp)):
		print('***Mode:', k, ', Damping:', round(damp2_comp[k], 3), ', Freq:', round(f2_comp[k], 3), ', ', round(ev2_comp[k], 4))
		for i in range(len(x2_comp)):
			p_f2_comp[i, k] = np.abs(vr2_comp[i, k]) * np.abs(w2_comp[k, i]) / np.max(np.abs(vr2_comp[:, k] * w2_comp[k, :]))
			if p_f2_comp[i, k] > threshold:
				print(states_pss[i], ':', round(p_f2_comp[i, k], 2), sep='', end=' ')
		print('\n')
	d["G2"]=np.round(p_f2_comp[0, :], 2)
	d["G3"]=np.round(p_f2_comp[8, :], 2)
	d["G4"]=np.round(p_f2_comp[16, :], 2)
	df2_comp = pd.DataFrame(d)
	df2_comp.to_csv('Type2_PSS_response.csv')

	E = np.zeros((J1.shape[0], 1))
	E[4, 0] = dps.Ka[0]/dps.Ta[0]
	F = np.zeros((1, J1.shape[0]))
	F[0, 1] = 1
	E_comp = np.zeros((J1_comp.shape[0], 1))
	E_comp[4, 0] = dps.Ka[0]/dps.Ta[0]
	F_comp = np.zeros((1, J1_comp.shape[0]))
	F_comp[0, 1] = 1
	sys1 = ctrl.StateSpace(J1, np.zeros((J1.shape[0], 1)), np.eye(J1.shape[0]), np.zeros((J1.shape[0], 1)))
	sys1_comp = ctrl.StateSpace(J1_comp, np.zeros((J1_comp.shape[0], 1)), np.eye(J1_comp.shape[0]), np.zeros((J1_comp.shape[0], 1)))
	Gs = ctrl.StateSpace(J1, E, F, [0])
	Gs_comp = ctrl.StateSpace(J1_comp, E_comp, F_comp, [0])
	s = ctrl.TransferFunction.s
	Gc = dps.K1*(s * dps.T1[0] + 1) / (s * dps.T2[0] + 1)
	Gw = (s * dps.Tw[0]) / (s * dps.Tw[0] + 1)
	# Bode plot for va2 w2 relationship
	# w_list = np.linspace(0.01, 120*pi*2, 500)
	plt.figure()
	ctrl.bode_plot(Gs, omega_num=400, omega_limits=(0.1, 2*pi*60))
	ctrl.bode_plot(Gc, omega_num=400, omega_limits=(0.1, 2*pi*60))
	ctrl.bode_plot(Gw, omega_num=400, omega_limits=(0.1, 2*pi*60))
	ctrl.bode_plot(Gs_comp, omega_num=400, omega_limits=(0.1, 2*pi*60))

	# ctrl.pzmap(Gs)
	# ctrl.pzmap(Gs_comp)
	# s = ctrl.tf('s')
	# G = np.linalg.det(s*np.eye(J1.shape[0]) - J1)


	# print(ev)
	plt.figure()
	plt.plot(ev1.real, ev1.imag, '.')
	plt.grid(True, which='both')
	plt.show()
	# plt.plot(ev2.real, ev2.imag, '.')
	# plt.grid(True, which='both')
	# plt.show()
	# plt.plot(ev3.real, ev3.imag, '.')
	# plt.grid(True, which='both')
	# plt.show()
	plt.plot(ev1_comp.real, ev1_comp.imag, '.')
	plt.grid(True, which='both')
	plt.show()
	#
	# fig = plt.figure()
	# ax = fig.add_subplot(111, polar=True)
	# ax.set_thetamin(0)
	# ax.set_thetamax(90)
	# angle = pi/2 - th1[0]
	# # print('ref angle: ', angle*180/pi)
	# plt.polar([angle, angle], [0, 1], label="ref")
	#
	# angle = pi/2 - (th1[0] - d[1])
	# # print('v angle: ', angle*180/pi)
	# plt.polar([angle, angle], [0, v[1]], label="v")
	# angle = np.angle(Edp1[0] + 1j*Eqp1[0])
	# # print('E angle: ', angle*180/pi)
	# mag = np.abs(Edp1[0] + 1j*Eqp1[0])
	# plt.polar([angle, angle], [0, mag], label="E")
	#
	#
	# ax.legend()
	#
	# plt.show()
	A = dps.A_dyn(x1, y, type1['vref'], type1['Pc'])
	B = dps.B_dyn(x1, y, type1['vref'], type1['Pc'])
	C = dps.C_dyn(x1, y)
	D = dps.D_dyn(x1, y)
	A_pss = dps.A_dyn_comp(x1_comp, y, type1['vref'], type1['Pc'])
	B_pss = dps.B_dyn_comp(x1_comp, y, type1['vref'], type1['Pc'])
	C_pss = dps.C_dyn(x1_comp, y)
	D_pss = dps.D_dyn(x1_comp, y)
	print('type 1 min damping w/PSS:', np.min(damp1_comp))
	print('type 2 min damping w/PSS:', np.min(damp2_comp))
	print('end')

