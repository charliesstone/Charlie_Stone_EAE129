import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import inspect as ispc
residual = 5e-10

##    --------      Charlie Stone EAE 129 Midterm Project       --------    ##
##    --------                SSID: 920605938                   --------    ##
##    --------             Please open in VSCode!               --------    ##
##    --------   Requires TeX Distribution to display matplot   --------    ##

#region dataframe ini
dneg5dat1 = {
    "AoA": np.linspace(-5, 30, 8),
    "C_L": [0.0580, 0.3800, 0.7020, 1.0240, 1.3460, 1.6680, 1.9900, 2.3120]
}
CLvA_dneg5 = pd.DataFrame(dneg5dat1)

d0dat1 = {
    "AoA": np.linspace(-5, 30, 8),
    "C_L": [0.0780, 0.4, 0.722, 1.044, 1.366, 1.688, 2.01, 2.3320]
}
CLvA_d0 = pd.DataFrame(d0dat1)

d5dat1 = {
    "AoA": np.linspace(-5, 30, 8),
    "C_L": [0.0980, 0.42, 0.742, 1.064, 1.386, 1.708, 2.03, 2.352]
}
CLvA_d5 = pd.DataFrame(d5dat1)

dneg5dat2 = {
    "AoA": np.linspace(-5, 30, 8),
    "C_M": [0.3385, 0.31, 0.2815, 0.2530, 0.2245, 0.196, 0.1675, 0.139]
}
CMvA_dneg5 = pd.DataFrame(dneg5dat2)

d0dat2 = {
    "AoA": np.linspace(-5, 30, 8),
    "C_M": [0.2785, 0.25, 0.2215, 0.1930, 0.1645, 0.1360, 0.1075, 0.0790]    
}
CMvA_d0 = pd.DataFrame(d0dat2)

d5dat2 = {
    "AoA": np.linspace(-5, 30, 8),
    "C_M": [0.2185, 0.19, 0.1615, 0.1330, 0.1045, 0.0760, 0.0475, 0.0190]     
}
CMvA_d5 = pd.DataFrame(d5dat2)
#endregion#

#region plot construct
# initialization #
plt.rcParams["text.usetex"] = True

# C_L vs AoA #
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(CLvA_dneg5["AoA"], CLvA_dneg5["C_L"], label=r"$\delta_e$ = -5 deg")
ax1.plot(CLvA_d0["AoA"], CLvA_d0["C_L"], label=r"$\delta_e$ = 0 deg")
ax1.plot(CLvA_d5["AoA"], CLvA_d5["C_L"], label=r"$\delta_e$ = 5 deg")
ax1.set_xlabel(r"Angle of Attack, $\alpha$ [deg]")
ax1.set_ylabel(r"Aircraft Coefficient of Lift, $C_L$")
ax1.set_title(r"$C_L$ vs $\alpha$ for selected Elevator Deflection Angles $\delta_e$")
ax1.legend()
ax1.grid(True)
# plt.xlim(0,5)
# plt.ylim(0.5, 0.75)

# C_L vs AoA, zoomed in #
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(CLvA_dneg5["AoA"], CLvA_dneg5["C_L"], label=r"$\delta_e$ = -5 deg")
ax2.plot(CLvA_d0["AoA"], CLvA_d0["C_L"], label=r"$\delta_e$ = 0 deg")
ax2.plot(CLvA_d5["AoA"], CLvA_d5["C_L"], label=r"$\delta_e$ = 5 deg")
ax2.set_xlabel(r"Angle of Attack, $\alpha$ [deg]")
ax2.set_ylabel(r"Aircraft Coefficient of Lift, $C_L$")
ax2.set_title(r"$C_L$ vs $\alpha$ for selected Elevator Deflection Angles $\delta_e$ (Zoomed in for Clarity)")
ax2.legend()
ax2.grid(True)
plt.xlim(0,5)
plt.ylim(0.5, 0.75)

# C_M vs AoA #
fig3, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(CMvA_dneg5["AoA"], CMvA_dneg5["C_M"], label=r"$\delta_e$ = -5 deg")
ax3.plot(CMvA_d0["AoA"], CMvA_d0["C_M"], label=r"$\delta_e$ = 0 deg")
ax3.plot(CMvA_d5["AoA"], CMvA_d5["C_M"], label=r"$\delta_e$ = 5 deg")
ax3.set_xlabel(r"Angle of Attack, $\alpha$ [deg]")
ax3.set_ylabel(r"Aircraft Coefficient of Moment about CG, $C_M$")
ax3.set_title(r"$C_M$ vs $\alpha$ for selected Elevator Deflection Angles $\delta_e$")
ax3.legend()
ax3.grid(True)

# C_M vs C_L #
fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.plot(CLvA_dneg5["C_L"], CMvA_dneg5["C_M"], label=r"$\delta_e$ = -5 deg")
ax4.plot(CLvA_d0["C_L"], CMvA_d0["C_M"], label=r"$\delta_e$ = 0 deg")
ax4.plot(CLvA_d5["C_L"], CMvA_d5["C_M"], label=r"$\delta_e$ = 5 deg")
ax4.set_xlabel(r"Aircraft Coefficient of Lift, $C_L$")
ax4.set_ylabel(r"Aircraft Coefficient of Moment about CG, $C_M$")
ax4.set_title(r"$C_M$ vs $C_L$ for selected Elevator Deflection Angles $\delta_e$")
ax4.legend()
ax4.grid(True)
#endregion

#region aero/stab d/dx
# initialization of CL #
CLdcoeffdat1 = {
    "C_L0": np.linspace(1,1,3),
    "C_La": [CLvA_dneg5.loc[0, "AoA"], CLvA_d0.loc[4, "AoA"], CLvA_d5.loc[7, "AoA"]],
    "C_Lde": np.linspace(-5, 5, 3)
}
CLdcoeff1 = pd.DataFrame(CLdcoeffdat1)
CLdcoeffdat2 = {
    "C_L0": np.linspace(1,1,3),
    "C_La": [CLvA_dneg5.loc[1, "AoA"], CLvA_d0.loc[5, "AoA"], CLvA_d5.loc[6, "AoA"]],
    "C_Lde": np.linspace(-5, 5, 3)
}
CLdcoeff2 = pd.DataFrame(CLdcoeffdat2)

CLdat1 = {
    "C_L": [CLvA_dneg5.loc[0, "C_L"], CLvA_d0.loc[4, "C_L"], CLvA_d5.loc[7, "C_L"]]
}
CLvec1 = pd.DataFrame(CLdat1)
CLdat2 = {
    "C_L": [CLvA_dneg5.loc[1, "C_L"], CLvA_d0.loc[5, "C_L"], CLvA_d5.loc[6, "C_L"]]
}
CLvec2 = pd.DataFrame(CLdat2)

# initialization of CM #
CMdcoeffdat1 = {
    "C_M0": np.linspace(1,1,3),
    "C_Ma": [CMvA_dneg5.loc[0, "AoA"], CMvA_d0.loc[4, "AoA"], CMvA_d5.loc[7, "AoA"]],
    "C_Mde": np.linspace(-5, 5, 3)
}
CMdcoeff1 = pd.DataFrame(CMdcoeffdat1)
CMdcoeffdat2 = {
    "C_L0": np.linspace(1,1,3),
    "C_Ma": [CMvA_dneg5.loc[1, "AoA"], CMvA_d0.loc[5, "AoA"], CMvA_d5.loc[6, "AoA"]],
    "C_Mde": np.linspace(-5, 5, 3)
}
CMdcoeff2 = pd.DataFrame(CMdcoeffdat2)

CMdat1 = {
    "C_M": [CMvA_dneg5.loc[0, "C_M"], CMvA_d0.loc[4, "C_M"], CMvA_d5.loc[7, "C_M"]]
}
CMvec1 = pd.DataFrame(CMdat1)
CMdat2 = {
    "C_M": [CMvA_dneg5.loc[1, "C_M"], CMvA_d0.loc[5, "C_M"], CMvA_d5.loc[6, "C_M"]]
}
CMvec2 = pd.DataFrame(CMdat2)

# solving system of equations #
def equation_solver(dcoeff1, vec1, dcoeff2, vec2):
    """given 2 seperate matricies of coefficients and 2 seperate vector RHS 
    of a series of equations, the function attempts to solve each system independently
    and compare values. If the values match, the function produces the solution."""

    A = dcoeff1.to_numpy()
    b = vec1.to_numpy()
    A2 = dcoeff2.to_numpy()
    b2 = vec2.to_numpy()
    if np.linalg.det(A) == 0:
        current_line = ispc.currentframe().f_lineno
        print(f"Line {current_line}: A1 matrix is noninvertable, choose different sample values")
    if np.linalg.det(A2) == 0:
        current_line = ispc.currentframe().f_lineno
        print(f"Line {current_line}: A2 matrix is noninvertable, choose different sample values")
    else:
        Ainv = np.linalg.inv(A)
        A2inv = np.linalg.inv(A2)
        x = Ainv @ b
        x2 = A2inv @ b2
        if sum(x - x2) > residual:
            print("Values of Stability Derivatives do not match. Please troubleshoot.")
        else:
            print("Values of Stability Derivatives match. Initializing:")
    deriv_data = pd.DataFrame(x, columns = ["Values:"], index = dcoeff1.columns)
    print(deriv_data)
    return deriv_data


# execution #
CLderiv_data = equation_solver(CLdcoeff1, CLvec1, CLdcoeff2, CLvec2)
CMderiv_data = equation_solver(CMdcoeff1, CMvec1, CMdcoeff2, CMvec2)
#endregion

#region static margin
SMneg5 = - (CMvA_dneg5.loc[1, ["C_M"]] - CMvA_dneg5.loc[4, ["C_M"]]).item() / (CLvA_dneg5.loc[1, ["C_L"]] - CLvA_dneg5.loc[4, ["C_L"]]).item()
SM0 = - (CMvA_d0.loc[1, ["C_M"]] - CMvA_d0.loc[4, ["C_M"]]).item() / (CLvA_d0.loc[1, ["C_L"]] - CLvA_d0.loc[4, ["C_L"]]).item()
SM5 = - (CMvA_d5.loc[1, ["C_M"]] - CMvA_d5.loc[4, ["C_M"]]).item() / (CLvA_d5.loc[1, ["C_L"]] - CLvA_d5.loc[4, ["C_L"]]).item()
if np.abs(SMneg5 - SM0) + np.abs(SMneg5 - SM5) + np.abs(SM0 - SM5) > residual:
    print("SM values do not match. Please select different reference values.")
else:
    print("Values of Static Margin match. Initializing:")
    SM = np.round(np.average([SMneg5, SM0, SM5]), 3)
    print(f"Static Margin (SM) = {SM}")
#endregion

#region detrim
# function definition #
def detrim(C_M0, SM, C_L, C_Mde):
    """The function takes input arguments to the elevator trim angle formula,
    producing a value of detrim based on the input arguments. It assumes C_M0, SM, 
    and C_Mde are scalar values, while C_L is a column of a pandas DataFrame."""
    C_Lval = C_L.to_numpy()
    detrim = - (C_M0 - SM * C_Lval) / (C_Mde)
    return detrim

# plot generation #
detrim_val = detrim(
    CMderiv_data.loc["C_M0", "Values:"].item(), 
    SM, 
    CLvA_d0["C_L"], 
    CMderiv_data.loc["C_Mde", "Values:"].item()
    )
fig5, ax5 = plt.subplots(figsize=(10, 6))
ax5.plot(CLvA_d0["AoA"], detrim_val)
ax5.set_xlabel(r"Trim Angle of Attack, $\alpha$ trim [deg]")
ax5.set_ylabel(r"Elevator Trim Angle of Deflection $\delta_e$ trim")
ax5.set_title(r"Elevator Trim Angle versus Trim Angle of Attack")
#endregion

#region neutral point
# defining necessary variables #
C_la = 0.11 #[/deg]
eta_h = 1.0
de_da = 0
ARw = 1.676
ARh = 3.82
e = 1.0
Sw = 150 #[in^2]
Sh = 30 #[in^2]
xac_w = 0.25
xac_h = 4.0

# function for C_La #
def C_la_to_C_La(C_la, AR, e):
    """The function takes C_la (coefficient of lift of the lifting surface's airfoil),
    AR (aspect ratio of the lifting surface), and e (oswald efficiency factor) and 
    computes the finite-wing lift coefficient C_La."""
    C_La = C_la / (1 + (57.3* C_la)/(np.pi * e * AR))
    return C_La

# execution #
C_Law = C_la_to_C_La(C_la, ARw, e)
C_Lah = C_la_to_C_La(C_la, ARh, e)

# function for B #
def Bfunc(C_Law, C_Lah, eta_h, Sh, Sw, de_da):
    """The function takes C_Law (finite wing coefficient of lift), 
    C_Lah (finite tail coefficient of lift), 
    eta_h (tail efficieny factor), Sh (tail area), Sw (wing area),
    and de/da (change in downwash angle wrt angle of attack), and 
    computes coefficient factor B, which is used in the computation
    of aerodynamic center/neutral point of the aircraft."""
    B = C_Lah/C_Law * eta_h * Sh/Sw * (1 - de_da)
    return B

# execution #
B = Bfunc(C_Law, C_Lah, eta_h, Sh, Sw, de_da)

# function for neutral point #
def neutralpoint(x_acw, x_ach, B):
    """The function takes the aerodynamic centers of the wing and tail, and,
    along with the coefficient factor B, it calculates the aerodyanmic center 
    of the entire aircraft."""
    x_ac = (x_acw + B * x_ach)/(1 + B)
    return x_ac

# execution #
x_ac = np.round(neutralpoint(xac_w, xac_h, B), 3)
print(f"Using C_la = 0.11/deg, Neutral Point of the Aircraft lies at x_np = {x_ac}")

#region final commands
plt.show()

#endregion