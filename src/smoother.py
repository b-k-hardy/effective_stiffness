# Smoothing functions for effective stiffness analysis
import itertools
from typing import Literal

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class DisplacementSmoother:
    """Class for smoothing displacements on a mesh."""

    def __init__(
        self,
        mesh: pv.PolyData,
        n_neighbors: int = 1,
        weighting_method: Literal["uniform", "squared"] = "uniform",
    ) -> None:
        self.mesh = mesh
        self.residuals = []

        # Initialize the laplacian matrix. Guess what: (FUCK) this thing needs to be updating
        # After each major iteration... (not the iterative method's subiterations with relax factor)
        self.laplacian_matrix = DisplacementSmoother._build_combinatorial_laplacian(
            mesh,
            n_neighbors,
            weighting_method,
        )

    @staticmethod
    def _build_combinatorial_laplacian(
        mesh: pv.PolyData,
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

        return L.tocsr()

    def smoothing_loop(
        self,
        max_iter: int = 100,
        weight: float = 1000.0,
    ) -> None:
        """Loop over the smoothing process."""
        # This method can be used to iterate over the smoothing process
        # and apply the smoothing to the mesh points.
        self.residuals.clear()  # clear residuals if you'd like to re-run with different weighting

        if self.smoothing_method == "matrix_solve":
            smoothed_mesh = self.matrix_laplacian_smoothing(
                self.mesh,
                weight=self.laplacian,
                n_neighbors=1,
                weighting_method="uniform",
            )
        elif self.smoothing_method == "iterative":
            # some extra work needs to be done here since I screwed up the laplacian builder

            smoothed_mesh = self.iterative_matrix_laplacian_smoothing(relaxation=0.1)

    def smoother_analysis(self):
        """Perform analysis on the smoother."""

    def matrix_laplacian_smoothing(self, weight: float = 1000.0) -> pv.PolyData:
        """Perform Laplacian smoothing on a mesh using a matrix approach."""
        V = self.mesh.points.copy()
        n = V.shape[0]
        L = self.laplacian_matrix

        # Solve LᵀL V = 0 using normal equations
        # A = L.T @ L + sp.diags([weight] * mesh.n_points) # NOTE: trying to remove the transpose
        A = L + sp.diags([weight] * self.mesh.n_points)
        b = weight * V

        # Unconstrained smoothing (not recommended—mesh may collapse)
        V_new = np.zeros_like(self.mesh.points)
        for dim in range(3):
            V_new[:, dim] = spla.spsolve(A, b[:, dim])

        smoothed_mesh = self.mesh.copy()
        smoothed_mesh.points = V_new
        return smoothed_mesh


def iterative_matrix_laplacian_smoothing(self, relaxation: float = 0.1) -> pv.PolyData:
    V = self.mesh.points.copy()
    n = V.shape[0]

    L = self.laplacian_matrix
    S = sp.diags(np.ones(n)) - relaxation * L

    # Unconstrained smoothing (not recommended—mesh may collapse)
    V_new = np.zeros_like(V)
    for dim in range(3):
        V_new[:, dim] = spla.spsolve(S, np.zeros(n))

    V_new = S @ V
    smoothed_mesh = self.mesh.copy()
    smoothed_mesh.points = V_new
    return smoothed_mesh


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
