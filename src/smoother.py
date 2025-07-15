# Smoothing functions for effective stiffness analysis
import itertools
from typing import Literal

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class DisplacementSmoother:
    """Class for smoothing displacements on a mesh."""

    # Big thing: this class structure will make it easier to store things that are needed for the smoother,
    # such as the connectivity of the mesh etc... (I think?)
    def __init__(self, mesh: pv.PolyData):
        self.mesh = mesh
        self.laplacian = DisplacementSmoother._build_combinatorial_laplacian(mesh, 0, 1, weighting_method="uniform")

    @staticmethod
    def _build_combinatorial_laplacian(
        mesh: pv.PolyData,
        relaxation: float,
        n_neighbors: int,
        weighting_method: Literal["uniform", "squared"] = "uniform",
    ) -> sp.csr_matrix:
        n = len(mesh.points)
        rows, cols, diag, data = [], [], [], []

        # Loop over all edges
        for i in range(len(mesh.points)):
            inner_data = []
            neighbor_points_idx = mesh.point_neighbors_levels(i, n_neighbors)
            neighbor_points_idx = list(itertools.chain.from_iterable(neighbor_points_idx))
            for j in neighbor_points_idx:
                rows.append(i)
                cols.append(j)

                if weighting_method == "squared":
                    inner_data.append(-1 / (np.linalg.norm(mesh.points[i] - mesh.points[j] + 0.05) ** 2))

            if weighting_method == "uniform":
                diag.append(len(neighbor_points_idx))
            elif weighting_method == "squared":
                data.extend(inner_data)
                diag.append(-np.sum(inner_data))

        # Off-diagonal entries: -1 for each neighbor
        if weighting_method == "uniform":
            data = -np.ones(len(rows))
        L = sp.coo_matrix((data, (rows, cols)), shape=(n, n))

        # Diagonal: degree of each vertex
        degrees = np.array(diag)
        L += sp.diags(degrees)
        if relaxation < 1e-7:
            # If no relaxation, return the combinatorial Laplacian
            return L.tocsr()

        S = sp.diags(np.ones(n)) - relaxation * L

        return S.tocsr()

    def smooth(self, n_iter: int = 10, n_neighbors: int = 1) -> pv.PolyData:
        """Smooth the mesh displacements."""
        return iterative_laplacian_smoothing(self.mesh, n_iter, n_neighbors)


def iterative_laplacian_smoothing(
    mesh: pv.PolyData,
    n_iter: int = 10,
    n_neighbors: int = 1,
) -> pv.PolyData:
    """Perform iterative Laplacian smoothing on a mesh."""
    V = mesh.points.copy()
    n = V.shape[0]

    for _ in range(n_iter):
        V_new = V.copy()
        for i in range(n):
            neighbor_points_idx = mesh.point_neighbors_levels(i, n_neighbors)
            neighbor_points_idx = list(itertools.chain.from_iterable(neighbor_points_idx))
            if not neighbor_points_idx:
                continue
            V_new[i] = V[neighbor_points_idx].mean(axis=0)
        V = V_new

    smoothed_mesh = mesh.copy()
    smoothed_mesh.points = V
    return smoothed_mesh


def matrix_laplacian_smoothing(
    mesh: pv.PolyData,
    weight: float = 1000.0,
    n_neighbors: int = 1,
    weighting_method: Literal["uniform", "squared"] = "uniform",
) -> pv.PolyData:
    """Perform Laplacian smoothing on a mesh using a matrix approach."""
    V = mesh.points.copy()
    n = V.shape[0]
    L = _build_combinatorial_laplacian(mesh, 0, n_neighbors, weighting_method=weighting_method)

    # Solve LᵀL V = 0 using normal equations
    # A = L.T @ L + sp.diags([weight] * mesh.n_points) # NOTE: trying to remove the transpose
    A = L + sp.diags([weight] * mesh.n_points)
    b = weight * V

    # Unconstrained smoothing (not recommended—mesh may collapse)
    V_new = np.zeros_like(mesh.points)
    for dim in range(3):
        V_new[:, dim] = spla.spsolve(A, b[:, dim])

    smoothed_mesh = mesh.copy()
    smoothed_mesh.points = V_new
    return smoothed_mesh


def iterative_matrix_laplacian_smoothing(
    mesh: pv.PolyData,
    relaxation: float = 0.1,
    n_neighbors: int = 1,
) -> pv.PolyData:
    V = mesh.points.copy()
    n = V.shape[0]

    L = _build_combinatorial_laplacian(mesh, relaxation, n_neighbors)

    # Solve LᵀL V = 0 using normal equations
    # A = L.T @ L
    A = L
    # Handle constraints
    # if fixed_idx is not None and len(fixed_idx) > 0:
    #    free_idx = np.setdiff1d(np.arange(n), fixed_idx)

    #    A_ff = A[free_idx][:, free_idx]
    #    A_fc = A[free_idx][:, fixed_idx]
    #    V_fixed = V[fixed_idx]

    # RHS is -A_fc @ V_fixed for each coordinate (x, y, z)
    #    V_new = V.copy()
    #    for dim in range(3):
    #        rhs = -A_fc @ V_fixed[:, dim]
    #        V_new[free_idx, dim] = spla.spsolve(A_ff, rhs)

    # Unconstrained smoothing (not recommended—mesh may collapse)
    V_new = np.zeros_like(V)
    for dim in range(3):
        V_new[:, dim] = spla.spsolve(A, np.zeros(n))

    V_new = L @ V
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
