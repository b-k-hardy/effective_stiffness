# Smoothing functions for effective stiffness analysis
import itertools

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _build_combinatorial_laplacian(mesh: pv.PolyData):
    n = len(mesh.points)
    rows, cols = [], []

    # Loop over all edges
    for i in range(len(mesh.points)):
        neighbors = mesh.point_neighbors(i)
        for j in neighbors:
            rows.append(i)
            cols.append(j)

    # Off-diagonal entries: -1 for each neighbor
    data = -np.ones(len(rows))
    L = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

    # Diagonal: degree of each vertex
    degrees = np.array([len(mesh.point_neighbors(i)) for i in range(n)])
    L += sp.diags(degrees)

    return L.tocsr()


def laplacian_smoothing(mesh: pv.PolyData, fixed_idx: np.ndarray = None) -> pv.PolyData:
    V = mesh.points.copy()
    n = V.shape[0]

    L = _build_combinatorial_laplacian(mesh)

    # Solve LᵀL V = 0 using normal equations
    A = L.T @ L

    # Handle constraints
    if fixed_idx is not None and len(fixed_idx) > 0:
        free_idx = np.setdiff1d(np.arange(n), fixed_idx)

        A_ff = A[free_idx][:, free_idx]
        A_fc = A[free_idx][:, fixed_idx]
        V_fixed = V[fixed_idx]

        # RHS is -A_fc @ V_fixed for each coordinate (x, y, z)
        V_new = V.copy()
        for dim in range(3):
            rhs = -A_fc @ V_fixed[:, dim]
            V_new[free_idx, dim] = spla.spsolve(A_ff, rhs)

    # Unconstrained smoothing (not recommended—mesh may collapse)
    else:
        V_new = np.zeros_like(V)
        for dim in range(3):
            V_new[:, dim] = spla.spsolve(A, np.zeros(n))

    mesh.points = V_new
    return mesh


def fxn(x):
    # distance function to use with map()... if that makes sense
    return None


def laplacian_smoothing_weighted(
    mesh: pv.PolyData,
    weight: float,
    n_iter: int = 100,
    n_neighbors: int = 1,
) -> pv.PolyData:
    # construct matrix for laplacian smoothing...
    laplacian_matrix = np.zeros((mesh.n_points, mesh.n_points))
    # loop over rows
    for i in range(mesh.n_points):
        # find the neighbors of the point depending on the n_neighbors parameter
        neighbor_points_idx = mesh.point_neighbors_levels(i, n_neighbors)
        neighbor_points_idx = list(itertools.chain.from_iterable(neighbor_points_idx))

        # might be able to use map() or something efficient to also find the length from the original point to the neighbors
        # and then use that to weight the laplacian matrix

        # when done just put results in correct spot in row...

    neighbor_points_idx = np.array(neighbor_points_idx)

    return None
