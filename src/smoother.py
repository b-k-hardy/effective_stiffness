"""Iterative Laplacian Smoothing for Meshes."""

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
        diastolic_mesh: pv.PolyData,
        systolic_mesh: pv.PolyData,
        n_neighbors: int = 1,
        weighting_method: Literal["uniform", "squared"] = "uniform",
    ) -> None:
        """Initialize the displacement smoother.

        This method initializes the smoother with the combinatorial Laplacian matrix for the first pass.

        Args:
            diastolic_mesh (pv.PolyData): The diastolic mesh to smooth.
            systolic_mesh (pv.PolyData): The systolic mesh to align with.
            n_neighbors (int, optional): The number of neighbors to consider for smoothing. Defaults to 1.
            weighting_method (Literal["uniform", "squared"], optional): The weighting method to use. Defaults to "uniform".

        """
        self.mesh = diastolic_mesh.copy()  # make a copy of the mesh to avoid modifying the original
        self.systolic_mesh = systolic_mesh
        self.residuals = []
        self.weighting_method = weighting_method
        self.n_neighbors = n_neighbors

        # Initialize the laplacian matrix. Guess what: (FUCK) this thing needs to be updating
        # After each major iteration... (not the iterative method's subiterations with relax factor)
        self.laplacian_matrix = self._build_combinatorial_laplacian()

    def _build_combinatorial_laplacian(self) -> sp.csr_matrix:
        n = len(self.mesh.points)
        rows, cols, diag, data = [], [], [], []

        # Loop over all edges
        for i in range(len(self.mesh.points)):
            inner_data = []
            neighbor_points_idx = self.mesh.point_neighbors_levels(i, self.n_neighbors)
            neighbor_points_idx = list(itertools.chain.from_iterable(neighbor_points_idx))
            for j in neighbor_points_idx:
                rows.append(i)
                cols.append(j)

                if self.weighting_method == "squared":
                    inner_data.append(-1 / (np.linalg.norm(self.mesh.points[i] - self.mesh.points[j] + 0.05) ** 2))

            if self.weighting_method == "uniform":
                diag.append(len(neighbor_points_idx))
            elif self.weighting_method == "squared":
                data.extend(inner_data)
                diag.append(-np.sum(inner_data))

        # Off-diagonal entries: -1 for each neighbor
        if self.weighting_method == "uniform":
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
    ) -> pv.PolyData:
        """Loop over the smoothing process."""
        # This method can be used to iterate over the smoothing process
        # and apply the smoothing to the mesh points.
        self.residuals.clear()  # clear residuals if you'd like to re-run with different weighting
        self.residuals.append(np.inf)  # initialize with a large value

        while len(self.residuals) - 1 < max_iter + 1 and self.residuals[-1] > atol:
            old_points = self.mesh.points.copy()
            intermediate_mesh = self.inner_laplacian_smoothing(weight=weight)
            _, closest_points = self.systolic_mesh.find_closest_cell(
                intermediate_mesh.points,
                return_closest_point=True,
            )
            self.mesh.points = closest_points
            # Calculate the residuals
            distance = np.linalg.norm(old_points - self.mesh.points, axis=1)
            residual = np.linalg.norm(distance)
            print(f"Iteration {len(self.residuals) - 1}: ||x^(n-1) - x^n|| = {residual:.6f}")
            self.residuals.append(residual)
            # has to be rebuilt after each iteration if we do anything other than uniform weighting
            if self.weighting_method != "uniform":
                self.laplacian_matrix = self._build_combinatorial_laplacian()

        self.residuals = np.array(self.residuals[1:])  # remove the initial inf value
        return self.mesh

    def inner_laplacian_smoothing(self, weight: float = 1000.0) -> pv.PolyData:
        """Perform Laplacian smoothing on a mesh using a matrix solve approach."""
        V = self.mesh.points.copy()
        n = V.shape[0]
        L = self.laplacian_matrix

        # Solve the system
        # A = L.T @ L + sp.diags([weight] * self.mesh.n_points)  # NOTE: trying to remove the transpose
        A = L + sp.diags([weight] * self.mesh.n_points)
        b = weight * V
        V_new = np.zeros_like(self.mesh.points)
        for dim in range(3):
            V_new[:, dim] = spla.spsolve(A, b[:, dim])

        inter_mesh = self.mesh.copy()
        inter_mesh.points = V_new

        return inter_mesh
