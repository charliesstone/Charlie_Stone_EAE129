import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import scipy.signal as sig
np.set_printoptions(suppress=True, precision=6)

## Charlie Stone 920605939 
## EAE 129 Final Project
## Please open in VSCode for proper initialization of regions

#region full ss def
# constants and parameters (for Mach 0.257 and sea level, F-104A)
u0 = 286.92  # reference airspeed (ft/s) at Mach 0.257 and sea level
g = 32.174  # gravitational acceleration (ft/s^2)
MAC = 9.55  # mean aerodynamic chord (ft)
Iy = 58611.0  # moment of inertia about the pitch axis (slug-ft^2)
m = 16300.0 / g  # mass of the aircraft (slug)
rho = 0.002377  # air density at sea level (slug/ft^3)
Q = 0.5 * rho * u0**2  # dynamic pressure (lb/ft^2)
S = 196.1  # wing reference area (ft^2)
S_t = 50 # tail reference area (ft^2)
tao = 0.5 # elevator efficiency
eta = 0.95 # dynamic pressure ratio


# aerodynamic coefficients (from page 253, F-104A, Mach 0.257 and sea level)
C_D0 = 0.263  # zero-lift drag coefficient
C_Da = 0.45 # drag - slope coefficient
C_Du = 0.0  # H.O.T (Assumed to be zero)
C_L0 = 0.735  # lift coefficient at zero AoA
C_Lu = 0.0  # H.O.T (Assumed to be zero)
C_La = 3.44  # lift - slope coefficient
C_ma = -0.64  # pitching moment coefficient derivative w.r.t. AoA
C_mu = 0 # H.O.T (Assumed to be zero)
C_mq = -5.8  # pitching moment coefficient derivative w.r.t. pitch rate
C_ma_dot = -1.6  # pitching moment coefficient derivative w.r.t. AoA rate
C_lat = 3 # lift - slope coefficient for tail

# control derivatives (elevator deflection)
C_Zde = - C_lat * tao * eta * S_t/S # z-force coefficient derivative w.r.t. elevator deflection
C_mde = -1.46  # pitching moment coefficient derivative w.r.t. elevator deflection

# compute stability derivatives using equations from Table 4.2
X_u = -(C_Du + 2 * C_D0) * (Q * S) / (m * u0)
X_w = -((C_Da - C_L0) * Q * S)/(m * u0)
X_de = 0
Z_u = -(C_Lu + 2 * C_L0) * (Q * S) / (m * u0)
Z_w = -((C_La + C_D0) * Q * S)/(m * u0)
Z_de = - C_Zde * Q * S / m
M_u = C_mu * (Q * S * MAC) / (Iy * u0)
M_w = C_ma * (Q * S * MAC)/(u0 * Iy)
M_w_dot = C_ma_dot * (MAC/(2 * u0)) * (Q * S * MAC)/(u0 * Iy)
M_de = (C_mde * Q * S * MAC)/Iy
M_q = C_mq * (Q * S * MAC**2) / (Iy * 2 * u0)

A = np.array([
    [X_u, X_w, 0, -g],
    [Z_u, Z_w, u0, 0],
    [M_u + M_w_dot * Z_u, M_w + M_w_dot * Z_w, M_q + M_w_dot * u0, 0],
    [0,0,1,0]
])

B = np.array([
    [X_de],
    [Z_de],
    [M_de + M_w_dot * Z_de],
    [0]
])

print(f"A matrix: {A}")
print(f"B matrix: {B}")
#endregion

#region full ss eig/freq
#generating ss
C = np.eye(4)
D = np.zeros((4,1))
ss = sig.StateSpace(A, B, C, D)
eig, _ = np.linalg.eig(ss.A)
speig = [np.round(eig[0], 6), np.round(eig[1], 6)]
lpeig = [np.round(eig[2], 6), np.round(eig[3], 6)]

#complex plot
plt.figure(1)
plt.scatter([np.real(eig[0]), np.real(eig[1])], [np.imag(eig[0]), np.imag(eig[1])], marker= 'x', c= "black", label= "Short Period mode")
plt.scatter([np.real(eig[2]), np.real(eig[3])], [np.imag(eig[2]), np.imag(eig[3])], marker= 'x', c= "blue", label= "Phugoid Mode")
plt.axhline(0, color='black', linewidth=0.5, linestyle='-')
plt.axvline(0, color='black', linewidth=0.5, linestyle='-')
plt.xlabel("real axis")
plt.ylabel("imaginary axis")
plt.title("Eigenvalues of F104A Longitudinal System")
plt.legend()
# plt.savefig("D:/School/classes/EAE129/finalproj/eigplot.png") #replace with plt.show() if examing source
plt.close()

#system attribute calculations
spcoeffs = np.poly(speig)
spnatfreq = np.sqrt(spcoeffs[2])
spdampratio = spcoeffs[1]/(2 * spnatfreq)
spdampfreq = np.imag(speig[0])
sptimecnst = -1 / (np.real(speig[0]))

lpcoeffs = np.poly(lpeig)
lpnatfreq = np.sqrt(lpcoeffs[2])
lpdampratio = lpcoeffs[1]/(2 * lpnatfreq)
lpdampfreq = np.imag(lpeig[0])
lptimecnst = -1 / (np.real(lpeig[0]))
#endregion

#region full ss sim
time = np.linspace(0, 200, 1000)
ufree = np.zeros_like(time)
x0free = np.array([0, 0, 0, 0.1])
y, t, x = sig.lsim(ss, ufree, time, x0free)
plt.figure(2)
state_labels = [
    r"$\Delta u$, Perturbations in Forward Velocity", 
    r"$\Delta w$, Perturbations in Vertical Velocity", 
    r"$\Delta q$, Perturbations in Pitch Rate", 
    r"$\Delta \theta$, Perturbations in Pitch Angle"
]
state_units = ["ft/s", "ft/s", "deg/s", "deg"]
state_colors = ["red", "orange", "blue", "green"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Time (Free) Response of Longitudinal Dynamic System")
for i, ax in enumerate(axes.flat):
    if i < 2:
        ax.plot(time, x[:, i], label= state_labels[i], c= state_colors[i])
    else: 
        ax.plot(time, np.rad2deg(x[:, i]), label= state_labels[i], c= state_colors[i])
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Time, t [sec]")
    ax.set_ylabel(f"Response [{state_units[i]}]")
    ax.set_title(state_labels[i])
    ax.grid()
plt.tight_layout()
# plt.savefig("D:/School/classes/EAE129/finalproj/freeresponse.png")
plt.close()

x0step = np.zeros([1, 4])
ustep = np.ones_like(time)
y, t, x = sig.lsim(ss, ustep, time, x0step)
plt.figure(3)
state_labels = [
    r"$\Delta u$, Perturbations in Forward Velocity", 
    r"$\Delta w$, Perturbations in Vertical Velocity", 
    r"$\Delta q$, Perturbations in Pitch Rate", 
    r"$\Delta \theta$, Perturbations in Pitch Angle"
]
state_units = ["ft/s", "ft/s", "deg/s", "deg"]
state_colors = ["red", "orange", "blue", "green"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Step Response of Longitudinal Dynamic System")
for i, ax in enumerate(axes.flat):
    if i < 2:
        ax.plot(time, x[:, i], label= state_labels[i], c= state_colors[i])
    else: 
        ax.plot(time, np.rad2deg(x[:, i]), label= state_labels[i], c= state_colors[i])
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel("Time, t [sec]")
    ax.set_ylabel(f"Response [{state_units[i]}]")
    ax.set_title(state_labels[i])
    ax.grid()
plt.tight_layout()
# plt.savefig("D:/School/classes/EAE129/finalproj/stepresponse.png")
plt.close()
#endregion

#region approx ss def
A_sp = np.array([
    [Z_w, u0],
    [M_w + M_w_dot * Z_w, M_q + M_w_dot * u0]
])

A_lp = np.array([
    [X_u, -g],
    [-Z_u/u0, 0]
])

B_approx = np.zeros([2, 1])
C_approx = np.array([1, 0])
D_approx = 0

print(f"SP Approx  A: {A_sp}")
print(f"LP Approx  A: {A_lp}")
spapprox_ss = sig.StateSpace(A_sp, B_approx, C_approx, D_approx)
lpapprox_ss = sig.StateSpace(A_lp, B_approx, C_approx, D_approx)
approx_speig, _ = np.linalg.eig(spapprox_ss.A)
approx_lpeig, _ = np.linalg.eig(lpapprox_ss.A)

approx_spcoeffs = np.poly(approx_speig)
approx_spnatfreq = np.sqrt(approx_spcoeffs[2])
approx_spdampratio = approx_spcoeffs[1]/(2 * approx_spnatfreq)
approx_spdampfreq = np.imag(approx_speig[0])
approx_sptimecnst = -1 / (np.real(approx_speig[0]))

approx_lpcoeffs = np.poly(approx_lpeig)
approx_lpnatfreq = np.sqrt(approx_lpcoeffs[2])
approx_lpdampratio = approx_lpcoeffs[1]/(2 * approx_lpnatfreq)
approx_lpdampfreq = np.imag(approx_lpeig[0])
approx_lptimecnst = -1 / (np.real(approx_lpeig[0]))
#endregion

#region outputs
print(f"short period characteristic polynomial coefficients: {approx_spcoeffs}")
print(f"phugoid characteristic polynomial coefficients: {approx_lpcoeffs}")
# Print comparison of exact and approximate short-period mode
print("Short-Period Mode Comparison:")
print(f"Exact Eigenvalues: {speig}, Approximate Eigenvalues: {approx_speig}")
print(f"Exact Natural Frequency: {spnatfreq:.4f} rad/s, Approximate Natural Frequency: {approx_spnatfreq:.4f} rad/s")
print(f"Exact Damping Ratio: {spdampratio:.4f}, Approximate Damping Ratio: {approx_spdampratio:.4f}")
print(f"Exact Damped Frequency: {spdampfreq:.4f} rad/s, Approximate Damped Frequency: {approx_spdampfreq:.4f} rad/s")
print(f"Exact Time Constant: {sptimecnst:.4f} s, Approximate Time Constant: {approx_sptimecnst:.4f} s")

# Print comparison of exact and approximate phugoid mode
print("Phugoid Mode Comparison:")
print(f"Exact Eigenvalues: {lpeig}, Approximate Eigenvalues: {approx_lpeig}")
print(f"Exact Natural Frequency: {lpnatfreq:.4f} rad/s, Approximate Natural Frequency: {approx_lpnatfreq:.4f} rad/s")
print(f"Exact Damping Ratio: {lpdampratio:.4f}, Approximate Damping Ratio: {approx_lpdampratio:.4f}")
print(f"Exact Damped Frequency: {lpdampfreq:.4f} rad/s, Approximate Damped Frequency: {approx_lpdampfreq:.4f} rad/s")
print(f"Exact Time Constant: {lptimecnst:.4f} s, Approximate Time Constant: {approx_lptimecnst:.4f} s")
#endregion