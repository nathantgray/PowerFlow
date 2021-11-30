"""
Pg2 = Vd2 * Id2 + Vq2 * Iq2
Vd2 = V2 * sin(th2 - d2)
Vq2 = V2 * cos(th2 - d2)
Iq2 = -1/x'q2 * [E'd2 - V2 * sin(th2 - d2)]
Id2 = 1/x'd2 * [E'q2 - V2 * cos(th2 - d2)]

Pg2 = V2 * sin(th2 - d2) * (1/x'd2 * [E'q2 - V2 * cos(th2 - d2)]) + V2 * cos(th2 - d2) * (-1/x'q2 * [E'd2 - V2 * sin(th2 - d2)])

Qg2 = V2 * cos(th2 - d2) * 1/x'd2 * [E'q2 - V2 * cos(th2 - d2)] + V2 * sin(th2 - d2) * (-1/x'q2 * [E'd2 - V2 * sin(th2 - d2)])

0 = Pg2  - V2 * V6 * 60 * cos(d2 - d6 - pi/2)
0 = Qg2  - V2**2 * 60 * sin(pi/2) - V2 * V6 * 60 * sin(d2 - d6 - pi/2)


Pg2 - PL2 = sum(Vi * Vj * Yij * cos(di - dj - thij))
Qg2 - QL2 = sum(Vi * Vj * Yij * sin(di - dj - thij))



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