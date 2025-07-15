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
        """Initialize the displacement smoother.

        This method initializes the smoother with a mesh and builds the combinatorial Laplacian matrix for the first pass.
        # NOTE: consider removing matrix builder and turn into a more dynamic approach

        Args:
            mesh (pv.PolyData): The input mesh.
            n_neighbors (int, optional): The number of neighbors to consider for smoothing. Defaults to 1.
            weighting_method (Literal["uniform", "squared"], optional): The weighting method to use. Defaults to "uniform".

        """
        self.orig_mesh = mesh  # Store the original mesh for reference
        self.mesh = mesh.copy()  # make a copy of the mesh to avoid modifying the original
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
        atol: float = 1e-6,
    ) -> None:
        """Loop over the smoothing process."""
        # This method can be used to iterate over the smoothing process
        # and apply the smoothing to the mesh points.
        self.residuals.clear()  # clear residuals if you'd like to re-run with different weighting

        while len(self.residuals) < max_iter and self.residuals[-1] > atol:
            smoothed_mesh = self.inner_laplacian_smoothing(
                self.mesh,
                weight=self.laplacian,
                n_neighbors=1,
                weighting_method="uniform",
            )

    def smoother_analysis(self):
        """Perform analysis on the smoother."""

    def inner_laplacian_smoothing(self, weight: float = 1000.0) -> pv.PolyData:
        """Perform Laplacian smoothing on a mesh using a matrix solve approach."""
        V = self.mesh.points.copy()
        n = V.shape[0]
        L = self.laplacian_matrix

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
