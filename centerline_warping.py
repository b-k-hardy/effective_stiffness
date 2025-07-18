import copy
import datetime
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.interpolate import make_splprep
from scipy.spatial.transform import Rotation as R

from src import smoother


class EffectiveStiffnessAnalysis:
    """Class to hold effective stiffness analysis data."""

    def __init__(self, patient_ID: str, data_paths: dict) -> None:
        # NOTE: need to figure out how to enable all general set up for the class so another analysis can be run modularly
        self.patient_ID = patient_ID
        self.raw_centerline_diastole = pv.read(data_paths["centerline_diastole"])
        self.raw_centerline_systole = pv.read(data_paths["centerline_systole"])
        self.full_surface_diastole = pv.read(data_paths["full_surface_diastole"])
        self.full_surface_systole = pv.read(data_paths["full_surface_systole"])
        self.centerline_spline_diastole: pv.PolyData | None = None  # Placeholder for the fitted centerline spline
        self.centerline_spline_systole: pv.PolyData | None = None  # Placeholder for the fitted centerline spline

    # NOTE: try to make this a method with out of place modification?
    def fit_centerline_spline(
        self,
        n_spline_points: int = 500,
        smoothness: float = 10,
        reverse: tuple = (False, False),
    ) -> Self:
        """Fit spline to the raw centerline points.

        This is applied to both diastolic and systolic centerlines.
        A single diastolic spline is returned with a displacement vector attribute (for systole)

        Args:
            centerline (pv.PolyData): _description_
            reverse (bool, optional): Decision on whether or not to reverse the points before fitting spline.
                                      Necessary since dumbass 3D Slicer vmtk doesn't consistently choose starting point.
                                      Defaults to False.

        Returns:
            Self

        """
        updated_analysis = copy.deepcopy(self)
        splines = []  # List to hold the fitted splines -- diastole first!
        for i, centerline in enumerate(
            [updated_analysis.raw_centerline_diastole, updated_analysis.raw_centerline_systole],
        ):
            points = centerline.points[::-1] if reverse[i] else centerline.points
            # FIXME: just make this thing use the topology or something... is that even possible or are we gonna have a bunch of fucked up tiny lines

            spline_function, u = make_splprep([points[:, 0], points[:, 1], points[:, 2]], s=smoothness)
            u = np.linspace(0, 1, n_spline_points)
            new_points = spline_function(u)
            first_deriv = spline_function(u, nu=1)

            new_points = np.array(new_points).T
            first_deriv = np.array(first_deriv).T

            spline_polydata = pv.PolyData(
                new_points,
                lines=[new_points.shape[0]] + np.arange(new_points.shape[0]).tolist(),
            )
            spline_polydata.point_data.set_vectors(first_deriv, "Derivative")
            spline_polydata = spline_polydata.compute_arc_length()
            splines.append(spline_polydata)

        centerline_displacement = splines[1].points - splines[0].points
        splines[0].point_data.set_vectors(centerline_displacement, "Displacement")
        updated_analysis.centerline_spline_diastole = splines[0]
        updated_analysis.centerline_spline_systole = splines[1]

        return updated_analysis

    def establish_zero_vector(self, cross_vector: np.ndarray = None) -> Self:
        updated_analysis = copy.deepcopy(self)
        if cross_vector is None:
            cross_vector = np.array([-1, 0, 0])

        centerline_splines = [updated_analysis.centerline_spline_diastole, updated_analysis.centerline_spline_systole]
        for i in range(len(centerline_splines)):
            centerline = centerline_splines[i]
            circumferential_zero_vectors = []
            for j in range(centerline.n_points):
                circ0 = np.cross(centerline["Derivative"][j], cross_vector)
                circumferential_zero_vectors.append(circ0 / np.linalg.norm(circ0))
            # NOTE: verify that these layers of references actually pass the data to the object in the end...
            centerline["zero_vector"] = np.array(circumferential_zero_vectors)
        return updated_analysis

    def circularity_analysis(self, clip_radius: float = 35.0):
        # function to perform circularity analysis on the centerline and full surface
        return -1

    def warp_by_centerline(self):
        return -1

    def add_surface_coordinates(self) -> Self:
        updated_analysis = copy.deepcopy(self)

        normalized_arc_length = (
            updated_analysis.full_surface_diastole["arc_length"]
            / updated_analysis.centerline_spline_diastole["arc_length"][-1]
        )
        circumferential_coordinate = (updated_analysis.full_surface_diastole["Circumferential Coordinate"] + np.pi) / (
            2 * np.pi
        )
        updated_analysis.full_surface_diastole.point_data.set_array(
            np.stack((normalized_arc_length, circumferential_coordinate), axis=1),
            name="surface_coordinates",
        )
        updated_analysis.full_surface_diastole = updated_analysis.full_surface_diastole.compute_derivative(
            scalars="surface_coordinates",
        )
        s_grad = updated_analysis.full_surface_diastole["gradient"][:, :3]
        theta_grad = updated_analysis.full_surface_diastole["gradient"][:, 3:]
        updated_analysis.full_surface_diastole.point_data.set_vectors(s_grad, name="s_gradient")
        updated_analysis.full_surface_diastole.point_data.set_vectors(theta_grad, name="theta_gradient")

        return updated_analysis

    # NOTE: copied from other script, need to refactor a bit

    # FIXME: need to split this function up... I don't want circularity shit in here!
    def centerline_nearest_neighbor_mapping(self) -> Self:
        """Run this function AFTER making centerline spline and finding circularity and arc length.

        Args:
            centerline_spline (pv.PolyData): _description_
            mesh (pv.PolyData): _description_

        """
        # loop through the mesh points and find the nearest centerline point. Take values at that point and assign to the mesh point.
        updated_analysis = copy.deepcopy(self)
        # mesh_circularity = np.zeros(mesh.n_points)
        mesh_arc_length = np.zeros(updated_analysis.full_surface_diastole.n_points)
        mesh_circ_coordinate = np.zeros(updated_analysis.full_surface_diastole.n_points)
        centerline_displacement = np.zeros((updated_analysis.full_surface_diastole.n_points, 3))
        additional_displacement = np.zeros((updated_analysis.full_surface_diastole.n_points, 3))
        # mesh_circularity.fill(np.nan)  # fill with NaN to make sure points that aren't assigned stand out when debugging
        mesh_arc_length.fill(np.nan)
        centerline_displacement.fill(np.nan)
        additional_displacement.fill(np.nan)

        for i, mesh_point in enumerate(updated_analysis.full_surface_diastole.points):
            # find the nearest centerline point
            centerline_points = updated_analysis.centerline_spline_diastole.points
            # calculate the distance from the node to each centerline point and find the index of the closest centerline point
            centerline_idx = np.argmin(np.linalg.norm(centerline_points - mesh_point, axis=1))
            # assign the circularity and arc length values from the centerline to the mesh point
            # mesh_circularity[i] = centerline_spline["Circularity"][centerline_idx]
            mesh_arc_length[i] = updated_analysis.centerline_spline_diastole["arc_length"][centerline_idx]
            circ_coordinate = mesh_point - updated_analysis.centerline_spline_diastole.points[centerline_idx]
            # project circ_coordinate onto the centerline derivative vector to get the circumferential coordinate
            circ_derivative = updated_analysis.centerline_spline_diastole["Derivative"][centerline_idx]
            circ_derivative /= np.linalg.norm(circ_derivative)  # normalize the derivative vector
            circ_coordinate = (
                circ_coordinate
                - np.dot(circ_coordinate, circ_derivative) / np.linalg.norm(circ_derivative) ** 2 * circ_derivative
            )
            circ_coordinate = np.arctan2(
                np.dot(
                    circ_coordinate,
                    np.cross(
                        circ_derivative,
                        updated_analysis.centerline_spline_diastole["zero_vector"][centerline_idx],
                    ),
                ),
                np.dot(circ_coordinate, updated_analysis.centerline_spline_diastole["zero_vector"][centerline_idx]),
            )
            mesh_circ_coordinate[i] = circ_coordinate
            centerline_displacement[i] = updated_analysis.centerline_spline_diastole["Displacement"][centerline_idx]
            additional_displacement[i] = (
                updated_analysis.full_surface_diastole["Displacement"][i]
                - updated_analysis.centerline_spline_diastole["Displacement"][centerline_idx]
            )

        # assign the circularity and arc length values to the mesh point data
        updated_analysis.full_surface_diastole.point_data.set_array(mesh_circ_coordinate, "Circumferential Coordinate")
        # mesh.point_data.set_array(mesh_circularity, "Circularity")
        updated_analysis.full_surface_diastole.point_data.set_array(mesh_arc_length, "arc_length")
        updated_analysis.full_surface_diastole.point_data.set_array(centerline_displacement, "Centerline Displacement")
        updated_analysis.full_surface_diastole.point_data.set_array(additional_displacement, "Additional Displacement")

        return updated_analysis

    def nearest_surface_neighbor_mapping(self) -> Self:
        updated_analysis = copy.deepcopy(self)
        mesh_d = updated_analysis.full_surface_diastole
        mesh_s = updated_analysis.full_surface_systole

        mesh_d = mesh_d.warp_by_vector("Centerline Displacement")
        # NOTE: holy fuck this is not symmetric... might need to take another look?
        closest_cells, closest_points = mesh_s.find_closest_cell(mesh_d.points, return_closest_point=True)
        displacements = closest_points - mesh_d.points
        mesh_d.point_data.set_vectors(
            displacements,
            "Nearest Surface Point Displacement",
        )
        mesh_d = mesh_d.warp_by_vector("Centerline Displacement", factor=-1.0)  # Reverse the warping direction

        # NOTE: A BUNCH OF CONFUSION ABOUT REFERNCES HERE... (AND EARLIER) MAYBE I HAVE TO DO THIS?
        updated_analysis.full_surface_diastole.point_data.set_vectors(
            displacements,
            "Nearest Surface Point Displacement",
        )

        return updated_analysis


def surface_ray_tracing_displacement(mesh_d: pv.PolyData, mesh_s: pv.PolyData) -> pv.PolyData:
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement")
    mesh_d = mesh_d.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

    mesh_d["Ray Tracing Displacement"] = np.empty((mesh_d.n_points, 3))
    for i in range(mesh_d.n_points):
        p = mesh_d.points[i]
        vec = mesh_d["Normals"][i] * 3
        p0 = p - vec
        p1 = p + vec
        ip, ic = mesh_s.ray_trace(p, p1 + vec, first_point=True)
        if ip.size == 0:
            ip, ic = mesh_s.ray_trace(
                p,
                p - vec,
                first_point=True,
            )  # Try the other direction if no intersection found... this actually doesn't make any sense given the original vector length but whatever...
        if ip.size == 0:
            print(f"Warning: No intersection found for point {i} at {p}.")
            ip = p
        # dist = np.sqrt(np.sum((ip - p) ** 2))
        mesh_d["Ray Tracing Displacement"][i] = ip - p

    # Replace zeros with nans
    mask = mesh_d["Ray Tracing Displacement"] == 0
    mesh_d["Ray Tracing Displacement"][mask] = np.nan
    # np.nanmean(h0n["distances"])

    mesh_d = mesh_d.warp_by_vector("Centerline Displacement", factor=-1.0)  # Reverse the warping direction
    return mesh_d


# NOTE: this can just be a static method since it doesn't depend on the class instance
def calc_circularity(area: float, perimeter: float) -> float:
    """Calculate the circularity of an aortic cross-section given its area and perimeter."""
    return 4 * np.pi * area / (perimeter**2)


def circularity_contours(
    full_surface: pv.PolyData,
    centerline_sp: pv.PolyData,
    clip_radius: float = 35.0,
) -> tuple[pv.PolyData, pv.PolyData, np.ndarray]:
    extracted_cuts = []
    extracted_surfaces = []
    circularity_array = []

    for i in range(centerline_sp.n_points):
        roi = pv.Sphere(center=centerline_sp.points[i], radius=clip_radius)

        test_slice = full_surface.slice(origin=centerline_sp.points[i], normal=centerline_sp["Derivative"][i])
        test_slice.clear_data()

        extracted = test_slice.clip_surface(roi, invert=True)
        extracted_surf = extracted.delaunay_2d()

        lengths = extracted.compute_cell_sizes()
        perimeter = np.sum(lengths["Length"])
        circularity = calc_circularity(extracted_surf.area, perimeter)
        circularity_array.append(circularity)

        extracted.cell_data.set_scalars(np.ones(extracted.n_cells) * perimeter, "Perimeter")
        extracted.cell_data.set_scalars(np.ones(extracted.n_cells) * circularity, "Circularity")

        extracted_cuts.append(extracted)
        extracted_surfaces.append(extracted_surf)

    contours = pv.merge(extracted_cuts)
    surfaces = pv.merge(extracted_surfaces)

    # NOTE: potentially return u from make_splprep to use for plotting... need to figure out what it means EXACTLY
    return contours, surfaces, np.array(circularity_array)


def main():
    CIRCULARITY_ANALYSIS = False  # Set to True if you want to plot circularity shifting and the unwrapped surface
    SAVE = True  # Set to True if you want to save the results. Good to disable during initial debugging to avoid overwriting files with junk.
    PLOT_RESIDUALS = False  # Set to True if you want to plot the residuals during the smoothing process
    # load the data
    vdm_id = "david_vdm"
    data_directory = f"data/{vdm_id}"
    output_directory = f"results/{vdm_id}"
    smoother_weighting_method = "squared"  # can be "uniform" or "squared"
    update_weight = 50.0
    max_iter = 100
    n_neighbor = 1
    atol = 0.5
    projection_type = "nearest"  # can be "nearest" or "ray_tracing"

    path_dictionary = {
        "centerline_diastole": f"{data_directory}/Centerline_model_David.vtp",
        "centerline_systole": f"{data_directory}/Centerline warped.vtp",
        "full_surface_diastole": f"{data_directory}/david_vdm.vtp",
        "full_surface_systole": f"{data_directory}/david_vdm_warped.vtp",
    }

    david_analysis = EffectiveStiffnessAnalysis("david", path_dictionary)
    david_analysis = david_analysis.fit_centerline_spline(reverse=(False, True))
    david_analysis = david_analysis.establish_zero_vector()
    david_analysis = david_analysis.centerline_nearest_neighbor_mapping()
    david_analysis = david_analysis.nearest_surface_neighbor_mapping()
    david_analysis = david_analysis.add_surface_coordinates()

    ##############################################################################################
    ## CIRCULARITY ANALYSIS CODE (commented out for now as I debug) ##
    """all_extracted_diastole, all_surfaces_diastole, circularity_diastole = circularity_contours(
        full_surface_diastole,
        centerline_diastole_spline,
        clip_radius=50.0,
    )
    all_extracted_systole, all_surfaces_systole, circularity_systole = circularity_contours(
        full_surface_systole,
        centerline_systole_spline,
        clip_radius=50.0,
    )

    centerline_diastole_spline["Circularity"] = (
        circularity_diastole  # only storing on the spline because of the dumbass way I've set up this function
    )
    centerline_systole_spline["Circularity"] = circularity_systole

    if SAVE:
        all_extracted_diastole.save(f"{output_directory}/david_diastole_contours.vtp")
        all_surfaces_diastole.save(f"{output_directory}/david_diastole_cross_sections.vtp")

        all_extracted_systole.save(f"{output_directory}/david_systole_contours.vtp")
        all_surfaces_systole.save(f"{output_directory}/david_systole_cross_sections.vtp")"""
    ##############################################################################################

    # Save the results
    if SAVE:
        david_analysis.centerline_spline_diastole.save(f"{output_directory}/david_spline_centerline_diastole_DEBUG.vtp")
        david_analysis.centerline_spline_systole.save(f"{output_directory}/david_spline_centerline_systole_DEBUG.vtp")
        david_analysis.full_surface_diastole.save(f"{output_directory}/david_diastole_surface_DEBUG.vtp")
        david_analysis.full_surface_systole.save(f"{output_directory}/david_systole_surface_DEBUG.vtp")

    # Smoothing and displacement analysis BELOW (for now -- putting it below initial saves)
    # Do full warping to the systolic state
    full_surface_diastole_warped = david_analysis.full_surface_diastole.warp_by_vector("Centerline Displacement")
    full_surface_diastole_warped = full_surface_diastole_warped.warp_by_vector("Nearest Surface Point Displacement")

    # NEW: try smoothing the displacement to see what happens. Compute derivative after smoothing SEPARATELY
    # NOTE: NEED TO SEE HOW DISPLACEMENT IS SMOOTHED AS WELL -- THIS IS WHAT WE ARE TARGETING BUT IDK IF IT MAKES ANY SENSE

    # My custom smoother class!
    custom_smoother = smoother.DisplacementSmoother(
        diastolic_mesh=full_surface_diastole_warped,
        systolic_mesh=david_analysis.full_surface_systole,
        n_neighbors=n_neighbor,
        weighting_method=smoother_weighting_method,
    )
    full_surface_diastole_warped_smoothed_test = custom_smoother.smoothing_loop(
        max_iter=100,
        weight=update_weight,
        atol=atol,
    )

    # compute the gradient of the surface coordinates after warping or warping+smoothing
    full_surface_diastole_warped = full_surface_diastole_warped.compute_derivative(scalars="surface_coordinates")
    s_grad = full_surface_diastole_warped["gradient"][:, :3]
    theta_grad = full_surface_diastole_warped["gradient"][:, 3:]
    full_surface_diastole_warped.point_data.set_vectors(s_grad, name="s_gradient")
    full_surface_diastole_warped.point_data.set_vectors(theta_grad, name="theta_gradient")

    full_surface_diastole_warped_smoothed_test = full_surface_diastole_warped_smoothed_test.compute_derivative(
        scalars="surface_coordinates",
    )
    s_grad_smoothed_test = full_surface_diastole_warped_smoothed_test["gradient"][:, :3]
    theta_grad_smoothed_test = full_surface_diastole_warped_smoothed_test["gradient"][:, 3:]
    full_surface_diastole_warped_smoothed_test.point_data.set_vectors(
        s_grad_smoothed_test,
        name="s_gradient",
    )
    full_surface_diastole_warped_smoothed_test.point_data.set_vectors(
        theta_grad_smoothed_test,
        name="theta_gradient",
    )

    if SAVE:
        full_surface_diastole_warped.save(f"{output_directory}/david_diastole_surface_warped_DEBUG.vtp")
        full_surface_diastole_warped_smoothed_test.save(
            f"{output_directory}/david_diastole_surface_warped_smoothed_test_{n_neighbor}_DEBUG.vtp",
        )

    # NEXT: need a way to associate the surface nodes with the centerline nodes... nearest neighbor for now? Then check in paraview
    # NOTE: I kind of do this already in a different script... just find that and copy?

    if PLOT_RESIDUALS:
        with plt.style.context(["science", "grid", "notebook"]):
            fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
            ax.set_title("Laplacian Smoothing Convergence")
            ax.plot(
                np.arange(len(custom_smoother.total_residuals)),
                custom_smoother.total_residuals,
                marker="o",
                markersize=3,
                label=r"Total: $\left\Vert \mathbf{x}^{n-1} - \mathbf{x}^{n} \right\Vert_2$",
            )
            ax.plot(
                np.arange(len(custom_smoother.smoothing_residuals)),
                custom_smoother.smoothing_residuals,
                marker="x",
                markersize=3,
                label=r"Smoothing: $\left\Vert \mathbf{x}^{n-1} - \mathbf{x}^{*,n} \right\Vert_2$",
            )
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Movement (mm)")
            # Create a textbox with custom_smoother parameters
            params_text = (
                f"DisplacementSmoother parameters:\n"
                f"n_neighbors: {n_neighbor}\n"
                f"weighting_method: {smoother_weighting_method}\n"
                f"weight: {update_weight}\n"
                f"max_iter: {max_iter}\n"
                f"atol: {atol}"
            )
            ax.text(
                0.95,
                0.95,
                params_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "alpha": 0.7},
            )
            ax.hlines(
                y=atol,
                xmin=0,
                xmax=len(custom_smoother.total_residuals) - 1,
                colors="r",
                linestyles="dashed",
                label=f"Tolerance: {atol:.2f}",
            )
            ax.legend(
                fontsize=10,
                loc="center right",
                framealpha=0.7,
            )
            # Save figure with date and time in filename
            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S")
            fig.savefig(f"{output_directory}/smoothing_tests/residuals_{timestamp}.png", dpi=300)

    """if CIRCULARITY_ANALYSIS:
        with plt.style.context(["science", "notebook"]):
            fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
            ax.set_title("Circularity of Aortic Cross-Sections")
            ax.plot(
                centerline_diastole_spline["arc_length"] / centerline_diastole_spline["arc_length"][-1],
                circularity_diastole,
                label=f"Diastole (length = {centerline_diastole_spline['arc_length'][-1]:.2f} mm)",
            )
            ax.plot(
                centerline_systole_spline["arc_length"] / centerline_systole_spline["arc_length"][-1],
                circularity_systole,
                label=f"Systole (length = {centerline_systole_spline['arc_length'][-1]:.2f} mm)",
            )
            ax.set_xlabel("Normalized Arc Length")
            ax.set_ylabel("Circularity")
            ax.legend()

        with plt.style.context(["science", "notebook"]):
            fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
            ax.set_title("Surface Unwrapped")
            ax.scatter(full_surface_diastole["Circumferential Coordinate"], full_surface_diastole["arc_length"])
            ax.set_xlabel("Circumferential Coordinate (radians)")
            ax.set_ylabel("Arc Length (mm)")

        plt.show()"""


if __name__ == "__main__":
    main()
