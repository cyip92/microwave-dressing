import numpy as _np

# Physical Parameters
hbar = 1.05457e-34
bohr_magneton = 9.274014e-24
mu_0 = 4e-7 * _np.pi
c = 299792458
electron_mass = 9.11e-31

# Atomic parameters
use_holmium = True
if use_holmium:
    HF_spacing = [0, 4.309, 9.405, 15.247, 21.788, 28.973, 36.741, 45.023]  # Spacing is in GHz
    g_F = [51/25., 39/25., 9/7., 39/35., 1., 23/25., 237/275., 9/11.]   # g factor for each set of F-levels
    g_I = -6.4e-4   # Nuclear g factor
    g_J = 1.19514  # Electronic g factor
    I = 7/2.    # Nuclear spin
    J = 15/2.    # Electron spin
    F_val = [4, 11]  # Start and end for F values
else:
    HF_spacing = [0, 6.834682]  # Spacing is in GHz
    g_F = [-1/2., 1/2.]   # g factor for each set of F-levels
    g_I = -9.951e-4     # Nuclear g factor
    g_J = 2.002331  # Electronic g factor
    I = 3/2.    # Nuclear spin
    J = 1/2.    # Electron spin
    F_val = [1, 2]  # Start and end for F values

num_states = sum([2*F + 1 for F in range(F_val[0], F_val[1] + 1)])

# Simulation parameters
intensity = 100    # Dressing field intensity in SI units (10 W/m2 = 1 mW/cm2)
detuning = -25e6     # Dressing field detuning (Hz)
field_freq = 5.096e9 + detuning    # Dressing field frequency (Hz)
photon_count = 0
start_field = 1e-4
end_field = 20001e-4
num_points = 1000
pi_polarization = True
