import math
import time
import Plotter
import numpy as np
import scipy.linalg as sp
import Hamiltonian_Constructor as HC
from tqdm import tqdm
from parameters import *


# Given an already-sorted eigenvalue list, attempts to find a magic field value between given states.  Takes in field
# values in any units and eigenvalues in SI, returns a list of pairs
# (magic field with same units of field_val, gap between states in GHz)
def find_magic_field(field_val, eig_val, s1, s2):
    gap = eig_val[s1] - eig_val[s2]
    dx = field_val[1] - field_val[0]
    sl = []
    mag_cond = []
    for x in range(len(gap) - 1):
        sl.append(gap[x+1] - gap[x])
    for x in range(3, len(sl) - 1):
        if (sl[x+1] * sl[x]) < 0:   # Negative product indicates a sign flip
            # Do a linear interpolation
            a = abs(sl[x] / (sl[x+1] - sl[x]))
            mag_cond.append((field_val[x] + a*dx,
                             1e-9 * abs((gap[x] + (a/dx) * (gap[x+1] - gap[x])) / (2 * math.pi * hbar))))
    return mag_cond


# Get the eigenvalues of a given Hamiltonian.  This is the part of the code where the imaginary part turns into B.
def get_unsorted_eigenvalues(h):
    xval = []
    eigs = []
    for B in tqdm(np.linspace(start_field, end_field, num_points), ascii=True):
        xval.append(B)
        step_eig = sp.eigvalsh(h.real + h.imag * B)
        eigs.append(step_eig)
    return xval, eigs


# Properly reorder the eigenvalues by calculating the current slope and picking the next point for each energy level
# based on linear extrapolation
def sort_eigenvalues(eig_val):
    eig_val[0].sort()      # Assume proper low-field ordering
    eig_val[1].sort()
    n_states = len(eig_val[0])
    for i in tqdm(range(len(eig_val) - 2), ascii=True):
        slopes = eig_val[i+1] - eig_val[i]
        tmp_val = eig_val[i+2]
        reordered = []
        for a in range(n_states):     # Find the closest points
            dist = float("inf")
            closest = -1
            for b in range(len(tmp_val)):
                match_dist = abs(tmp_val[b] - eig_val[i+1][a] - slopes[a])
                if match_dist < dist:
                    dist = match_dist
                    closest = b
            reordered.append(tmp_val[closest])
            tmp_val = np.delete(tmp_val, closest)
        for a in range(n_states):        # Assign the points AFTER all the comparisons
            eig_val[i+2][a] = reordered[a]
    return np.transpose(eig_val)


def print_delayed(string):
    """
    Print with a delay before and after to "fix" tqdm printing.

    :param string:  String to print.
    :return:        None
    """
    time.sleep(0.01)
    print string
    time.sleep(0.01)

state_list = HC.state_list
num_states = len(state_list)

# Construct and diagonalize this Hamiltonian.
print_delayed("Constructing Hamiltonian...")
H = HC.get_floquet_hamiltonian(photon_count, True)
H0 = HC.get_undressed_hamiltonian()
total_states = (2*photon_count + 1) * num_states

# Get all the eigenvalues as a function of the magnetic field.
print_delayed("Diagonalizing Floquet Hamiltonian (size = " + str(total_states) + ")...")
xval, all_eigs = get_unsorted_eigenvalues(H)

# Properly reorder the eigenvalues by calculating the current slope and picking the next point for each energy level
# based on linear extrapolation
print_delayed("Reordering Floquet eigenvalues...")
all_eigs = sort_eigenvalues(all_eigs)

# Redo it without state dressing to compare
print_delayed("Diagonalizing undressed Hamiltonian...")
xval0, eigs0 = get_unsorted_eigenvalues(H0)
print_delayed("Reordering undressed eigenvalues...")
eigs0 = sort_eigenvalues(eigs0)
dist = np.zeros((num_states, total_states))
for i_u in range(num_states):
    for i_d in range(total_states):
        dist[i_u][i_d] = np.sum(abs(eigs0[i_u] - all_eigs[i_d]))

# Pick out states closest to the undressed values and designate them as the dressed states
print_delayed("Matching dressed/undressed states...")
min_index = np.zeros(num_states)
for i_u in tqdm(range(num_states), ascii=True):
    lowest = float("inf")
    for i_d in range(total_states):
        if dist[i_u][i_d] < lowest:
            min_index[i_u] = i_d
            lowest = dist[i_u][i_d]
eigs = []
for i in range(num_states):
    eigs.append(all_eigs[int(min_index[i])])

# Set plot attributes
pol = r"$\pi$" if pi_polarization else r"$\bot$"
title = "Dressed Magic Fields on f = " + str((field_freq - detuning) / 1e9) + " GHz, $\Delta$ = "\
        + str(detuning / 1e6) + " MHz, and I = " + str(intensity / 10.) + " mW/cm$^2$ (" + pol + " polarization)"

# Make properly-scaled data for plotting (Energy in GHz vs. field in Gauss)
print_delayed("Rescaling data for plot...")
# plt.xlabel('Magnetic Field (G)')
pltX = []
for i in range(len(xval)):
    pltX.append(xval[i] * 1e4)
pltY = []
pltY_unshifted = []
for i in range(len(eigs)):
    pltY.append(eigs[i] / (hbar * 1e9 * 2 * math.pi))
    pltY_unshifted.append(eigs0[i] / (hbar * 1e9 * 2 * math.pi))

# Try to find magic fields
print "Searching for magic fields..."
l1 = {i for i in range(num_states) if state_list[i].M >= 0}
# mag_shifted = find_magic_field(pltX, eigs, s1, s2)
mag_list_unshifted = []
mag_list_shifted = []
for a in l1:
    l2 = {i for i in range(num_states) if (abs(state_list[i].M - state_list[a].M) <= 2 and
                                           abs(state_list[i].F - state_list[a].F) in [1, 2]) and
          state_list[i].M >= 0 and state_list[a].M >= 0}
    for b in l2:
        mag_unshifted = find_magic_field(pltX, eigs0, a, b)
        mag_shifted = find_magic_field(pltX, eigs, a, b)
        if mag_unshifted != [] and a < b:
            mag_list_unshifted.append([(a, b), mag_unshifted])
        if mag_shifted != [] and a < b:
            mag_list_shifted.append([(a, b), mag_shifted])
# Test : |4,1> (5) and |5,1> (15): 231 G -> 4.231 GHz, slope 2.03 MHz/G

# Various plotting examples
# Plotter.plot_animated_zeeman(state_list, eigs)
Plotter.plot_magic_fields("Unshifted Magic Fields", state_list, mag_list_unshifted)
# Plotter.plot_magic_fields(title, state_list, mag_list_shifted)
# Plotter.plot_breit_rabi("Holmium Breit-Rabi Diagram", xval, all_eigs, highlight=[s1, s2])
# Plotter.plot_gap(title, xval, eigs, s1, s2, other_eig=eigs0)
