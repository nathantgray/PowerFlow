import numpy as np
def PF_NewtonRaphson(d, V, Y, PQ, PV, Psched, Qsched, Qd, Qlim, prec, varargin):

# Uses Newton-Raphson method to solve the power-flow of a power system.
# Also capable of Q limiting.
# Written by Nathan Gray
# Arguments:
# d: list of voltage phase angles in system
# V: list of voltage magnitudes in system
# Y: Ybus matrix for system
# PQ: list of PQ busses
# PV: list of PV busses
# Psched, Qsched: list of real, reactive power injections
# Qd: Reactive power demand (Qd and Qlim are not used in this version)
# Qlim: array of Q limits- 1st column is maximums 2nd is minimums
# prec: program finishes when all mismatches < 10^-abs(prec)
# varargin: optional argument to specify maximum iterations
if ~isempty(varargin):
    # 'varargin' is an optional argument specifying the
    # maximum number of iterations.
    iterations = varargin{1}
else:
    # Default # iterations is 20.
    iterations = 10

N = np.shape(Y)[0]
#Ng = length(PV);

gMaxQ = []
gMinQ = []
## Newton Raphson
for it in range(iterations):
    # Calculate Jacobian
    J = PF_Jacobian_fast(PQ,Y,V,d)
    # Calculate Mismataches
    mis = mismatch(d,V,Y,PQ,PV, Psched, Qsched)
    # Calculate update values
    dx =J^-1*mis
    # Update angles: d_(n+1) = d_n + dd
    d(sort([PV,PQ])) = d(sort([PV,PQ])) + dx(1:N-1)
    # Update Voltages: V_(n+1) = V_n(1+dV/V_n)
    V(PQ) = V(PQ).*(1+dx(N:N+length(PQ)-1))
    # Check error
    mis = mismatch(d,V,Y,PQ,PV, Psched, Qsched)
    if max(mis) < 10^-abs(prec)
        print("Newton Raphson completed in %i iterations.\n",it)
        break


    if it>iterations-1
        print("Max iterations reached, %i.\n",it)



