import matplotlib.pyplot as plt
import Hamiltonian_Constructor as Hamiltonian
import math
from matplotlib.collections import LineCollection
from parameters import *


# Makes an energy level diagram with arrows joining states with magic field conditions.  The arrows are annotated with
# a box stating the value at which the magic B-field condition occurs.
def plot_magic_fields(title, state_list, mag_list):
    min_f = Hamiltonian.F_val[0]
    max_f = Hamiltonian.F_val[1]
    segments = LineCollection([[(m - 0.4, f), (m + 0.4, f)] for f in range(min_f, max_f + 1)
                                for m in range(f + 1)], linestyles='solid')
    axes = plt.axes()
    axes.set_xlim((-0.5, max_f + 0.5))
    axes.set_ylim((min_f - 0.5, max_f + 0.5))
    axes.add_collection(segments)
    for i in range(len(state_list)):    # State indexing labels for ease of use
        st = state_list[i]
        axes.annotate(str(st.index), xy=(st.M, st.F - 0.2), xycoords="data", va="center", ha="center", color="gray")
    for mag in mag_list:
        s1, s2 = mag[0]
        f1 = state_list[s1].F
        f2 = state_list[s2].F
        m1 = state_list[s1].M
        m2 = state_list[s2].M
        axes.annotate('', xy=(m1, f1), xycoords='data', xytext=(m2, f2),
                      textcoords='data', arrowprops=dict(arrowstyle="<->"))
        axes.annotate(str(round(mag[1][len(mag[1]) - 1][0], 4)), xy=((2*m1 + m2) / 3., (2*f1 + f2) / 3.),
                      xycoords="data", va="center", ha="center", bbox=dict(boxstyle="round", fc="w"))
    plt.xlabel("M")
    plt.ylabel("F")
    plt.title(title)
    plt.show()


# Makes a Breit-Rabi diagram from a given set of eigenvalues.  Takes in field and energy values in SI units, plots
# field in Gauss and energy in GHz.  Highlights specifiec fields.
def plot_breit_rabi(title, field_val, eig_val, highlight = None):
    plt_x = []
    for i in range(len(field_val)):
        plt_x.append(field_val[i] * 1e4)
    plt_y = []
    for i in range(len(eig_val)):
        plt_y.append(eig_val[i] / (hbar * 1e9 * 2 * math.pi))
    for i in range(len(eig_val)):
        plt.plot(plt_x, plt_y[i], '-' if i not in highlight else ':')
    plt.xlabel('Magnetic Field (Gauss)')
    plt.ylabel('Energy (GHz)')
    plt.title(title)
    plt.show()


# Plots the energy gap between two specified states in a properly-ordered set of Zeeman states.  Has the option to plot
# another one for comparison.
def plot_gap(title, field_val, eig_val, s1, s2, other_eig=None):
    plt_x = []
    for i in range(len(field_val)):
        plt_x.append(field_val[i] * 1e4)
    plt_y = []
    for i in range(len(eig_val)):
        plt_y.append(abs(eig_val[s1] - eig_val[s2]) / (hbar * 1e9 * 2 * math.pi))
        if other_eig is not None:
            plt_y.append(abs(other_eig[s1] - other_eig[s2]) / (hbar * 1e9 * 2 * math.pi))
    for i in range(len(eig_val)):
        plt.plot(plt_x, plt_y[i], '-')
    plt.xlabel('Magnetic Field (Gauss)')
    plt.ylabel('Energy (GHz)')
    plt.title(title)
    plt.show()


# Plots the difference between energy gaps between two specified states in a properly-ordered set of Zeeman states.
# This is primarily to determine the effect of state dressing.
def plot_shift(title, field_val, eig_val, s1, s2, other_eig):
    plt_x = []
    for i in range(len(field_val)):
        plt_x.append(field_val[i] * 1e4)
    plt_y = []
    for i in range(len(eig_val)):
        plt_y.append((abs(eig_val[s1] - eig_val[s2]) - abs(other_eig[s1] - other_eig[s2])) / (hbar * 2 * math.pi))
    for i in range(len(eig_val)):
        plt.plot(plt_x, plt_y[i], '-')
    plt.xlabel('Magnetic Field (Gauss)')
    plt.ylabel('Energy (Hz)')
    plt.title(title)
    plt.show()
