import matplotlib.pyplot as plt
import Hamiltonian_Constructor as Hamiltonian
import math
from matplotlib.collections import LineCollection
from parameters import *
import matplotlib.animation as animation


def plot_magic_fields(title, state_list, mag_list):
    """
    Makes an energy level diagram with arrows joining states with magic field conditions.  The arrows are annotated
    with a box stating the value at which the magic B-field condition occurs.

    :param title:       String to use as a title for the plot.
    :param state_list:  List of HFState objects to make labeling/positioning code easier to read.
    :param mag_list:    List of Lists; each entry is [(State_Index_A, State_Index_B), [Magic_Field_Info]] with A < B.
        Each sublist may contain multiple magic field conditions stored in Magic_Field_Info as (Field_Value, ...).
    :return:            None
    """
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


def plot_breit_rabi(title, field_val, eig_val, highlight=[]):
    """
    Makes a Breit-Rabi diagram from a given set of eigenvalues.  Plots field in Gauss and energy in GHz.

    :param title:       String to use as a title for the plot.
    :param field_val:   List of B-field values to use as x-values in the plot, assumed to be SI units.  List should
        be a list of numbers.
    :param eig_val:     List of energy values to use as y-values in the plot, assumed to be SI units. List should be
        such that each element is itself a list of energy values of the same length as field_val.
    :param highlight:   List of indices to highlight in the plot by using dotted lines instead of solid lines.
    :return:            None
    """
    plt_x = []
    for i in range(len(field_val)):
        plt_x.append(field_val[i] * 1e4)
    plt_y = []
    for i in range(len(eig_val)):
        plt_y.append(eig_val[i] / (hbar * 1e9 * 2 * math.pi))
    for i in range(len(eig_val)):
        plt.plot(plt_x, plt_y[i], ':' if i in highlight else '-')
    plt.xlabel('Magnetic Field (Gauss)')
    plt.ylabel('Energy (GHz)')
    plt.title(title)
    plt.show()


def plot_gap(title, field_val, eig_val, s1, s2, other_eig=None):
    """
    Plots the energy gap between two specified states in a properly-ordered set of Zeeman states.  Can optionally take
    another set of eigenvalues and plot the gap between the same states for comparison.

    :param title:       String to use as a title for the plot.
    :param field_val:   List of B-field values to use as x-values in the plot, assumed to be SI units.  List should
        be a list of numbers.
    :param eig_val:     List of energy values to use as y-values in the plot, assumed to be SI units. List should be
        such that each element is itself a list of energy values of the same length as field_val.
    :param s1:          Index of the first state to use for the gap.
    :param s2:          Index of the second state to use for the gap.
    :param other_eig:   Second list of energy values, formatted identically to eig_val.  If this is specified the plot
        will have two curves, one for the gap in eig_val and one for the gap in this.
    :return:            None
    """
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


def plot_shift(title, field_val, eig_val, s1, s2, other_eig):
    """
    Plots the difference between energy gaps between two specified states in a properly-ordered set of Zeeman states.
    This is primarily to determine the effect of state dressing.  Note that this function is essentiall the same as
    plot_gap() but requires a non-empty other_eig parameter and subtracts the two curves.

    :param title:       String to use as a title for the plot.
    :param field_val:   List of B-field values to use as x-values in the plot, assumed to be SI units.  List should
        be a list of numbers.
    :param eig_val:     List of energy values to use as y-values in the plot, assumed to be SI units. List should be
        such that each element is itself a list of energy values of the same length as field_val.
    :param s1:          Index of the first state to use for the gap.
    :param s2:          Index of the second state to use for the gap.
    :param other_eig:   Second list of energy values, formatted identically to eig_val.
    :return:            None
    """
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


def plot_animated_zeeman(state_list, eigs):
    """
    Plot an animated Zeeman effect figure, with x-axis being M and y-axis being energy in GHz.

    :param state_list:  List of HFState objects to make labeling/positioning code easier to read.
    :param eigs:        List of energy values to use as y-values in the plot, assumed to be SI units. List should be
        such that each element is itself a list of energy values of the same length as field_val.
    :return:            None
    """
    fig, axes = plt.subplots()

    # Determine the right scaling for the axes
    min_m = 0
    max_m = 0
    min_energy = 0
    max_energy = 0
    for state in state_list:
        min_m = min(min_m, state.M)
        max_m = max(max_m, state.M)
    for ind in range(len(eigs)):
        en = eigs[ind][len(eigs[0]) - 1] / (hbar * 1e9 * 2 * math.pi)
        min_energy = min(min_energy, en)
        max_energy = max(max_energy, en)
    spread = max_energy - min_energy
    axes.set_xlim((min_m - 0.5, max_m + 0.5))
    axes.set_ylim((min_energy - 0.05 * spread, max_energy + 0.05 * spread))

    plt.title("Holmium ground hyperfine states")
    plt.xlabel("$M_F$")
    plt.ylabel("Energy (GHz)")
    label = axes.text(min_m - 0.4, max_energy, '', fontsize=12)

    # This function is defined and then passed to FuncAnimation() to perform the animation.  i runs over the integers.
    def animate(i):
        state_lines = []
        for ind in range(len(eigs)):
            state = state_list[ind]
            en = eigs[ind][i] / (hbar * 1e9 * 2 * math.pi)
            state_lines.append([(state.M - 0.4, en), (state.M + 0.4, en)])
        col = LineCollection(state_lines, linestyles='solid')
        axes.add_collection(col)
        db = (end_field - start_field) / num_points
        label.set_text("B = " + str(1e4 * (start_field + i * db)) + " G")
        return col, label

    # This must be stored to a dummy variable to actually show the plot
    ani = animation.FuncAnimation(fig, animate, frames=num_points, interval=6000 / num_points, blit=True)
    plt.show()
