import math
import time
import Plotter
import numpy as np
import Hamiltonian_Constructor as HC
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
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


# Get the eigenvalues of a given Hamiltonian
def get_unsorted_eigenvalues(h):
    xval = []
    eigs = []
    for B in tqdm(np.linspace(start_field, end_field, num_points), ascii=True):
        xval.append(B)
        step_eig = np.linalg.eigvals(h.real + h.imag * B)
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


# Print with a delay before and after to "fix" tqdm printing
def print_delayed(string):
    time.sleep(0.01)
    print string
    time.sleep(0.01)

state_list = HC.state_list
num_states = len(state_list)

# Construct H_HFS, the Hamiltonian for the hyperfine splitting.
print_delayed("Constructing Hamiltonian...")
H_HFS = HC.get_hfs_hamiltonian()
H_Z = HC.get_zeeman_hamiltonian()
if pi_polarization:
    H_dress = HC.get_microwave_hamiltonian(intensity, 0)
else:
    H_dress = (HC.get_microwave_hamiltonian(intensity, 1) + HC.get_microwave_hamiltonian(intensity, -1)) / math.sqrt(2)
# rabi = field / (2 * math.pi * hbar) * bohr_magneton * mat_elem[2, 6]
# print "Rabi frequency is " + str(rabi) + " Hz"

# Construct the Floquet matrix (H_f) by first constructing a structured mask, then replacing elements in the mask with
# the actual sub-matrices
cpl = 1 * (np.eye(2*photon_count + 1, k=1) + np.eye(2*photon_count + 1, k=-1))
ph = 1j * np.diagflat(range(-photon_count, photon_count + 1))
hfs = 10 * np.eye(2*photon_count + 1)
mask = cpl + ph + hfs
H_f = []
total_states = (2*photon_count + 1) * num_states
for a in range(2*photon_count + 1):
    row = []
    for b in range(2*photon_count + 1):
        tmp = np.zeros((num_states, num_states))
        val = mask[a][b]
        if val.real == 1:
            tmp = H_dress
        elif val.real == 10:
            tmp = H_HFS + H_Z
        if val.imag != 0:
            tmp += np.eye(num_states) * val.imag * (hbar * 2 * math.pi * field_freq)
        if b == 0:
            row = tmp
        else:
            row = np.concatenate((row, tmp), axis=1)
    if a == 0:
        H_f = row
    else:
        H_f = np.concatenate((H_f, row), axis=0)

# Diagonalize this Hamiltonian.  Note that the values are theoretically strictly real, but the actual values are
# encoded such that real parts are B field-independent and imaginary parts are linearly dependent on the field.
H = H_f
H0 = H_HFS + H_Z      # Unshifted Hamiltonian

# Get all the eigenvalues as a function of the magnetic field.  Note the real/imag encoding stated above.
print_delayed("Diagonalizing Floquet Hamiltonian (size = " + str(len(H)) + ")...")
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
# print "Shifted conditions:   " + str(mag_shifted)
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
            # print " " + str(a) + " to " + str(b) + ": " + str(mag_unshifted)
            mag_list_unshifted.append([(a, b), mag_unshifted])
        if mag_shifted != [] and a < b:
            # print "*" + str(a) + " to " + str(b) + ": " + str(mag_shifted)
            mag_list_shifted.append([(a, b), mag_shifted])
# Test : |4,1> (5) and |5,1> (15): 231 G -> 4.231 GHz, slope 2.03 MHz/G

# Animated Zeeman plotting code
fig, axes = plt.subplots()
axes.set_xlim((-11.5, 11.5))
axes.set_ylim((-260, 320))
plt.title("Holmium ground hyperfine states")
plt.xlabel("$M_F$")
plt.ylabel("Energy (GHz)")
label = axes.text(-11, 290, '', fontsize=12)


def animate(i):
    state_lines = []
    for ind in range(len(eigs)):
        state = state_list[ind]
        en = eigs[ind][i] / (hbar * 1e9 * 2 * math.pi)
        state_lines.append([(state.M - 0.4, en), (state.M + 0.4, en)])
    col = LineCollection(state_lines, linestyles='solid')
    axes.add_collection(col)
    dB = (end_field - start_field) / num_points
    label.set_text("B = " + str(1e4 * (start_field + i*dB)) + " G")
    return col, label

ani = animation.FuncAnimation(fig, animate, frames=num_points, interval=6000/num_points, blit=True)
plt.show()

# Magic field plotting code
# Plotter.plot_magic_fields("Unshifted Magic Fields", state_list, mag_list_unshifted)
# Plotter.plot_magic_fields(title, state_list, mag_list_shifted)

s1 = 19
s2 = 31

# Breit-Rabi plotting code
# Plotter.plot_breit_rabi("Holmium Breit-Rabi Diagram", xval, all_eigs, highlight=[s1, s2])

# Energy gap plotting code
# Plotter.plot_gap(title, xval, eigs, s1, s2, other_eig=eigs0)
