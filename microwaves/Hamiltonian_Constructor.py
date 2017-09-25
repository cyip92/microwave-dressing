import math
import numpy as np
import sympy.physics.wigner as w
from parameters import *
import scipy.sparse as sp


class HFState:
    """
    HFState objects represent a single hyperfine state with easier to access attributes

    :attr F:        Quantum number F for the state.
    :attr M:        Quantum number M for the state.
    :attr index:    Designated index for the state, corresponds to row/column in Hamiltonian matrices.
    """
    def __init__(self, f_val, m_val, index_val):
        self.F = f_val
        self.M = m_val
        self.index = index_val

    def __str__(self):
        return "HFState (F,M) = (" + str(self.F) + "," + str(self.M) + ")"


# Creates a list containing all hyperfine states in order, with the appropriate attributes for usage in the rest of
# the calculation.  Ordered groups of increasing F, within which M runs from -F to F.
state_list = []
index = 0
for F in range(F_val[0], F_val[1]+1):
    for M in range(-F, F+1):
        state_list.append(HFState(F, M, index))
        index += 1


# Generic memoization class.
class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}

    def __call__(self, *args):
        if args not in self.memo:
            self.memo[args] = self.f(*args)
        return self.memo[args]


# Memoized 6j symbol function
@Memoize
def mem_6j(j_1, j_2, j_3, j_4, j_5, j_6):
    return w.wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6)


# Memoized Clebsch-Gordon function
@Memoize
def mem_cg(j_1, j_2, j_3, m_1, m_2, m_3):
    return w.clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3)


def matrix_element(i1, i2, q):
    """
    Calculate the gI*I + gJ*J matrix element, using indices as arguments.

    :param i1:  Index of the first state to use from state_list.
    :param i2:  Index of the second state to use from state_list.
    :param q:   Spherical tensor rank of the incoming field, should be in [-1, 0, 1]
    :return:    The value of < state_list[s1] | gI*I + gJ*J | state_list[s2] >
    """
    s1 = state_list[i1]
    s2 = state_list[i2]
    f1 = s1.F
    m1 = s1.M
    f2 = s2.F
    m2 = s2.M
    if abs(f1 - f2) > 1 or abs(m1 - m2) > 1:   # Selection rules cut out a lot of time
        return 0
    reduced = ((-1) ** (1 + I + J)) * math.sqrt(2*f1 + 1)
    elem_i = ((-1) ** f2) * math.sqrt(I * (I + 1) * (2*I + 1)) * mem_6j(I, J, f1, f2, 1, I)
    elem_j = ((-1) ** f1) * math.sqrt(J * (J + 1) * (2*J + 1)) * mem_6j(J, I, f1, f2, 1, J)
    elem = ((-1) ** q) * reduced * (g_I * elem_i + g_J * elem_j) * mem_cg(f1, 1, f2, m1, q, m2)
    return elem


def get_hfs_hamiltonian():
    """
    Constructs a hyperfine splitting Hamiltonian.

    :return:    A matrix containing on-diagonal energy values due to the hyperfine interaction.
    """
    h_hfs = np.zeros((num_states, num_states))
    for i in range(num_states):
        h_hfs[i, i] = hbar * 2 * math.pi * HF_spacing[state_list[i].F - F_val[0]] * 1e9
    return h_hfs


def get_zeeman_hamiltonian():
    """
    Constructs a Hamiltonian for the Zeeman effect, multiplied by i to denote a linear dependence on the B-field.

    :return:    A matrix containing terms arising due to a static external magnetic field mixing some states together
        in the |F, M_F> basis.  Contains both on-diagonal and off-diagonal elements.
    """
    h_z = np.empty((num_states, num_states), dtype=np.complex_)
    for a in range(num_states):
        for b in range(num_states):
            h_z[a, b] = bohr_magneton * matrix_element(a, b, 0) * 1j
    return h_z


def get_microwave_hamiltonian(intensity, pi_pol):
    """
    Constructs a Hamiltonian for the state dressing effect due to an incoming microwave-frequency field.

    :param intensity:   Value to use as the intensity of the microwave field, assumed to be in SI.
    :param pi_pol:      Polarization of the incoming field, either "pi" (parallel to external field) or perpendicular,
        which is an equal superposition of sigma+ and sigma-.
    :return:            A matrix containing terms arising due to the microwave field mixing some states together in
        the |F, M_F> basis.  Contains both on-diagonal and off-diagonal elements.
    """
    h_dress = np.empty((num_states, num_states))
    field = math.sqrt(2 * mu_0 * intensity / c)
    for a in range(num_states):
        for b in range(num_states):
            if pi_pol:
                h_dress[a, b] = 2 * math.pi * field * bohr_magneton * matrix_element(a, b, 0)
            else:
                h_dress[a, b] = 2 * math.pi * field * bohr_magneton * \
                                (matrix_element(a, b, 1) + matrix_element(a, b, -1)) / math.sqrt(2)
    # rabi = field / (2 * math.pi * hbar) * bohr_magneton * mat_elem[2, 6]
    # print "Rabi frequency is " + str(rabi) + " Hz"
    return h_dress


def get_undressed_hamiltonian():
    """
    Constructs the Hamiltonian for the hyperfine interaction and static magnetic field, but no dressing field.

    :return:    A matrix containing both on-diagonal and off-diagonal terms from both interactions.  Zeeman component
        is encoded in the imaginary part of each element.
    """
    return get_hfs_hamiltonian() + get_zeeman_hamiltonian()


def get_floquet_hamiltonian(photon_count, pi_pol):
    """
    Construct a Hamiltonian which incorporates the hyperfine interaction, Zeeman interaction, and microwave-field
    interaction.  It first makes a structured mask, from which it unpacks the individual elements into their respective
    matrices via a Floquet method.

    :param photon_count:    Number of photons specifying how many extra states to add to the Hamiltonian.
    :param pi_pol:          Boolean denoting the polarization of the dressing field.
    :return:                A matrix containing many more states due to interactions coupling the undressed states with
        virtual Floquet states.  Any dependence on the external magnetic field is stored in the imaginary part of each
        element.
    """
    # Construct the Floquet mask matrix
    cpl = 1 * (np.eye(2 * photon_count + 1, k=1) + np.eye(2 * photon_count + 1, k=-1))
    ph = 1j * np.diagflat(range(-photon_count, photon_count + 1))
    hfs = 10 * np.eye(2 * photon_count + 1)
    mask = cpl + ph + hfs

    h_f = []
    h_hfs = get_hfs_hamiltonian()
    h_z = get_zeeman_hamiltonian()
    h_dress = get_microwave_hamiltonian(intensity, pi_pol)
    for a in range(2 * photon_count + 1):
        row = []
        for b in range(2 * photon_count + 1):
            tmp = np.zeros((num_states, num_states))
            val = mask[a][b]
            if val.real == 1:
                tmp = h_dress
            elif val.real == 10:
                tmp = h_hfs + h_z
            if val.imag != 0:
                tmp += np.eye(num_states) * val.imag * (hbar * 2 * math.pi * field_freq)
            if b == 0:
                row = tmp
            else:
                row = np.concatenate((row, tmp), axis=1)
        if a == 0:
            h_f = row
        else:
            h_f = np.concatenate((h_f, row), axis=0)
    # return sp.bsr_matrix(h_f)
    return h_f
