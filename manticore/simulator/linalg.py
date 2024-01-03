# Python
from functools import lru_cache

# Math
import numpy as np
import pandas as pd
import galois as gl
GF2 = gl.GF(2)

def bin_matrix(arr, n, galois=False):
    arr = np.right_shift.outer(arr, np.arange(n-1, -1, -1)) & 1
    return GF2(arr) if galois else arr

@lru_cache(maxsize=None)
def bin_range(n):
    return bin_matrix(np.arange(2**n), n)

def unbin_matrix(arr):
    arr = np.asarray(arr, dtype=int)
    return arr @ (1 << np.arange(arr.shape[1]-1, -1, -1))

# Shifts the indices of frame to agree with the ordering of to_qubits and to_clbits
# If frame has qubits/clbits not present in to_qubits/to_clbits (i.e. the frame will shrink)
# the indices are returned such that the corresponding probabilities are marginalized
# over the removed qubits/clbits
def twiddle_indices(indices, from_qubits, from_clbits, to_qubits, to_clbits):
    M, N = len(to_clbits), len(to_qubits)

    # if 2 * N + M > 32:  # Redundant since always followed by FrameError check
    #     raise RuntimeError(f"cannot twiddle {frame} to more than 32 bits")

    carg_shifts = np.array([_index_safe(to_clbits, i) for i in from_clbits], dtype=int)
    qarg_shifts = np.array([_index_safe(to_qubits, i) for i in from_qubits], dtype=int)

    shifts = 1 << np.concatenate((
        2*N + M - 1 - carg_shifts,  # Classical part
        2*N - 1 - qarg_shifts,  # X part
        N - 1 - qarg_shifts  # Z part
    ))

    binary_indices = bin_matrix(indices, len(shifts))

    return np.bitwise_xor.reduce(binary_indices * shifts, axis=1)

def _index_safe(lst, x, default=0):
    try:
        return lst.index(x)
    except ValueError:
        return default
