import math
import numpy as np
import sympy.physics.wigner as w
from parameters import *


# HFState objects represent a single hyperfine state
class HFState:
    def __init__(self, f_val, m_val, index_val):
        self.F = f_val
        self.M = m_val
        self.index = index_val

    def __str__(self):
        return "HFState (F,M) = (" + str(self.F) + "," + str(self.M) + ")"

# Creates a list containing all hyperfine states in order, with the appropriate attributes.  Ordered groups of
# increasing F, within which M runs from -F to F.
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


# Memoized 6j function
@Memoize
def mem_6j(j_1, j_2, j_3, j_4, j_5, j_6):
    return w.wigner_6j(j_1, j_2, j_3, j_4, j_5, j_6)


# Memoized Clebsch-Gordon function
@Memoize
def mem_cg(j_1, j_2, j_3, m_1, m_2, m_3):
    return w.clebsch_gordan(j_1, j_2, j_3, m_1, m_2, m_3)


# Calculate the gI*I + gJ*J matrix element, using HFState objects as arguments.  q is the spherical tensor rank of the
# incoming field.
def matrix_element(s1, s2, q):
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


# Make a hyperfine splitting Hamiltonian
def get_hfs_hamiltonian():
    h_hfs = np.zeros((num_states, num_states))
    for i in range(num_states):
        h_hfs[i, i] = hbar * 2 * math.pi * HF_spacing[state_list[i].F - F_val[0]] * 1e9
    return h_hfs


# Construct H_Z, the Hamiltonian for the Zeeman effect.  Note that this is actually the Zeeman Hamiltonian divided by
# the magnetic field, multiplied by i to denote the DC field dependence.
def get_zeeman_hamiltonian():
    h_z = np.empty((num_states, num_states), dtype=np.complex_)
    for a in range(num_states):
        for b in range(num_states):
            h_z[a, b] = bohr_magneton * matrix_element(state_list[a], state_list[b], 0) * 1j
    return h_z


# Construct H_dress, the Hamiltonian for the microwave field state dressing.  q is the spherical tensor rank of the
# incoming field.
def get_microwave_hamiltonian(intensity, q):
    h_dress = np.empty((num_states, num_states))
    field = math.sqrt(2 * mu_0 * intensity / c)
    for a in range(num_states):
        for b in range(num_states):
            h_dress[a, b] = 2 * math.pi * field * bohr_magneton * matrix_element(state_list[a], state_list[b], q)
    return h_dress
