from collections import defaultdict
from more_itertools import pairwise

import numpy as np
import galois as gl
GF2 = gl.GF(2)

def row_echelon(mat, i=0, inplace=True):
    if not inplace:
        mat = mat.copy()
    
    m, n = mat.shape
    rows, cols = np.where(mat[i:, :]) # Find non-zero entries in non-eceheloned part

    if rows.size:
        j, k = rows[0]+i, cols[0] # Correct indices from line above

        # If the i-th row is all zero, swaps rows i and j
        if i != j:
            mat[[i, j]] = mat[[j, i]]

        # Add the i-th row to all rows below with a leading one
        for row in range(i+1, m):
            if mat[row, k]:
                mat[row, :] ^= mat[i, :]

        # Yield pivot column
        yield k
        yield from row_echelon(mat, i+1)

def smith_normal(mat, m, n, i=0):
    rows, cols = np.where(mat[i:m, i:n]) # Only check non-smithified part

    if rows.size:
        j, k = rows[0]+i, cols[0]+i # Correct index for cutting away i rows and columns above

        # Oh no. We found zero rows and/or columns above (j, k). Let us bring 1 at (j,k) to the top-left at (i, i)
        if i != j:
            mat[[i, j]] = mat[[j, i]] # Flips rows
        if i != k:
            mat[:, [i, k]] = mat[:, [k, i]] # Flips cols

        for row in range(i+1, m):
            # Zero out row
            if mat[row, i]: # If row has a leading 1
                mat[row, :] ^= mat[i, :] # Add row i to row
    
        for col in range(i+1, n):
            # Zero out column
            if mat[i, col]: # If col has a leading 1
                mat[:, col] ^= mat[:, i] # Add col i to col
        
        # We fixed one row and col. Recurse for the remaining matrix
        smith_normal(mat, m, n, i+1)

def snf(mat):
    m, n = mat.shape

    M = np.eye(m, dtype=int)
    N = np.eye(n, dtype=int)
    zeros = np.zeros((n, m), dtype=int)
    mat = GF2(np.block([[mat, M], [N, zeros]]))

    smith_normal(mat, m, n, 0)

    return mat[:m, :n], mat[:m, n:], mat[m:, :n]

def fundamental_subspaces(mat):
    A, P, Q = snf(mat)
    rank = np.count_nonzero(A)

    # Recall that A = P * mat * Q
    img = np.linalg.inv(P)[:, :rank]
    ker = Q[:, rank:]

    # Then A.T = Q.T * mat.T * P.T
    coimg = np.linalg.inv(Q.T)[:, :rank]
    coker = P.T[:, rank:]

    return dict(img=img, ker=ker, coimg=coimg, coker=coker)

def quotient(A, B):
    """Calculate generators of the quotient A/B

    Args:
        A (np.ndarray): An M x A matrix over GF2
        B (np.ndarray): An M x B matrix over GF2

    Returns:
        np.ndarray: An M x C matrix over GF2, where C = dim(A/B)
    """

    rank = B.shape[1]
    E = np.hstack((B, A))
    pivots = list(p for p in row_echelon(E, inplace=False) if p >= rank)

    return E[:, pivots]

def cartesian_transform(hom, bound):
    dim = hom.shape[1]
    basis_change = GF2.Zeros((dim, dim)) # Assumes a 3D crystal always has 3 logical operators
    
    for i in range(dim):
        tot = defaultdict(lambda: 0)
        for offset, mat in bound.items():
            offset_i = tuple(o if i != j else 0 for j, o in enumerate(offset))
            tot[offset_i] ^= mat @ hom

        mat = np.vstack(tuple(tot.values()))
        ker = fundamental_subspaces(mat)['ker']

        basis_change[:, i] = ker[:, 0] # Be careful. Can I assume the kernel is always dim 1?

    return hom @ basis_change

def boundary_maps(uc):
    elements = []
    for d in range(uc.dim + 1):
        elements.append(list(uc.filter_nodes(dim=d)))

    bounds = []
    for rows, cols in pairwise(elements):
        shape = (len(rows), len(cols))
        matrices = defaultdict(lambda: GF2.Zeros(shape))

        for u, v, offset in uc.edges(keys=True):
            if u in cols and v in rows:
                matrices[offset][rows.index(v), cols.index(u)] ^= 1

        bounds.append(dict(matrices))

    return bounds, elements
