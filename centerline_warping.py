import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scienceplots
from scipy.interpolate import make_splprep
from scipy.spatial.transform import Rotation as R


def surface_ray_tracing_displacement(mesh_d: pv.PolyData, mesh_s: pv.PolyData) -> pv.PolyData:
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement")
    mesh_d = mesh_d.compute_normals(point_normals=True, cell_normals=False, auto_orient_normals=True)

    mesh_d["Ray Tracing Displacement"] = np.empty((mesh_d.n_points, 3))
    for i in range(mesh_d.n_points):
        p = mesh_d.points[i]
        vec = (
            mesh_d["Normals"][i] * 3
        )  # 10 mm ray length... original example javi sent isn't good for closed surfaces...
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


def pyvista_surface_nearest_neighbor_mapping(mesh_d: pv.PolyData, mesh_s: pv.PolyData) -> pv.PolyData:
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement")
    closest_cells, closest_points = mesh_s.find_closest_cell(mesh_d.points, return_closest_point=True)
    displacements = closest_points - mesh_d.points
    mesh_d.point_data.set_vectors(
        displacements,
        "Nearest Surface Point Displacement",
    )
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement", factor=-1.0)  # Reverse the warping direction

    return mesh_d


# NOTE: copied from other script, need to refactor a bit
def centerline_nearest_neighbor_mapping(
    centerline_spline: pv.PolyData,
    mesh: pv.PolyData,
) -> None:
    """Run this function AFTER making centerline spline and finding circularity and arc length.

    Args:
        centerline_spline (pv.PolyData): _description_
        mesh (pv.PolyData): _description_

    """
    # loop through the mesh points and find the nearest centerline point. Take values at that point and assign to the mesh point.
    mesh_circularity = np.zeros(mesh.n_points)
    mesh_arc_length = np.zeros(mesh.n_points)
    mesh_circ_coordinate = np.zeros(mesh.n_points)
    centerline_displacement = np.zeros((mesh.n_points, 3))
    additional_displacement = np.zeros((mesh.n_points, 3))
    mesh_circularity.fill(np.nan)  # fill with NaN to make sure points that aren't assigned stand out when debugging
    mesh_arc_length.fill(np.nan)
    centerline_displacement.fill(np.nan)
    additional_displacement.fill(np.nan)

    for i, mesh_point in enumerate(mesh.points):
        # find the nearest centerline point
        centerline_points = centerline_spline.points
        # calculate the distance from the node to each centerline point and find the index of the closest centerline point
        centerline_idx = np.argmin(np.linalg.norm(centerline_points - mesh_point, axis=1))
        # assign the circularity and arc length values from the centerline to the mesh point
        mesh_circularity[i] = centerline_spline["Circularity"][centerline_idx]
        mesh_arc_length[i] = centerline_spline["arc_length"][centerline_idx]
        circ_coordinate = mesh_point - centerline_spline.points[centerline_idx]
        # project circ_coordinate onto the centerline derivative vector to get the circumferential coordinate
        circ_derivative = centerline_spline["Derivative"][centerline_idx]
        circ_derivative /= np.linalg.norm(circ_derivative)  # normalize the derivative vector
        circ_coordinate = (
            circ_coordinate
            - np.dot(circ_coordinate, circ_derivative) / np.linalg.norm(circ_derivative) ** 2 * circ_derivative
        )
        circ_coordinate = np.arctan2(
            np.dot(circ_coordinate, np.cross(circ_derivative, centerline_spline["zero_vector"][centerline_idx])),
            np.dot(circ_coordinate, centerline_spline["zero_vector"][centerline_idx]),
        )
        mesh_circ_coordinate[i] = circ_coordinate
        centerline_displacement[i] = centerline_spline["Displacement"][centerline_idx]
        additional_displacement[i] = mesh["Displacement"][i] - centerline_spline["Displacement"][centerline_idx]

    # assign the circularity and arc length values to the mesh point data
    mesh.point_data.set_array(mesh_circ_coordinate, "Circumferential Coordinate")
    mesh.point_data.set_array(mesh_circularity, "Circularity")
    mesh.point_data.set_array(mesh_arc_length, "arc_length")
    mesh.point_data.set_array(centerline_displacement, "Centerline Displacement")
    mesh.point_data.set_array(additional_displacement, "Additional Displacement")


def calc_circularity(area: float, perimeter: float) -> float:
    """Calculate the circularity of an aortic cross-section given its area and perimeter."""
    return 4 * np.pi * area / (perimeter**2)


def fit_spline(
    centerline: pv.PolyData,
    n_spline_points: int = 500,
    smoothness: float = 10,
    reverse: bool = False,
) -> pv.PolyData:
    """Fit spline to the centerline points.

    Args:
        centerline (pv.PolyData): _description_
        reverse (bool, optional): Decision on whether or not to reverse the points before fitting spline.
                                  Necessary since dumbass 3D Slicer vmtk doesn't consistently choose starting point.
                                  Defaults to False.

    Returns:
        pv.PolyData: _description_

    """
    points = centerline.points[::-1] if reverse else centerline.points
    # FIXME: just make this thing use the topology or something... is that even possible or are we gonna have a bunch of fucked up tiny lines

    spline_function, u = make_splprep([points[:, 0], points[:, 1], points[:, 2]], s=smoothness)
    u = np.linspace(0, 1, n_spline_points)
    new_points = spline_function(u)
    first_deriv = spline_function(u, nu=1)

    new_points = np.array(new_points).T
    first_deriv = np.array(first_deriv).T

    spline_polydata = pv.PolyData(new_points, lines=[new_points.shape[0]] + np.arange(new_points.shape[0]).tolist())
    spline_polydata.point_data.set_vectors(first_deriv, "Derivative")

    return spline_polydata


def circularity_contours(
    full_surface: pv.PolyData,
    centerline_sp: pv.PolyData,
    clip_radius: float = 35.0,
) -> tuple[pv.PolyData, pv.PolyData, np.ndarray, np.ndarray]:
    extracted_cuts = []
    extracted_surfaces = []
    circularity_array = []
    circumferential_zero_vectors = []

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

        circ0 = np.cross(centerline_sp["Derivative"][i], np.array([-1, 0, 0]))
        circumferential_zero_vectors.append(circ0 / np.linalg.norm(circ0))

    contours = pv.merge(extracted_cuts)
    surfaces = pv.merge(extracted_surfaces)

    # NOTE: potentially return u from make_splprep to use for plotting... need to figure out what it means EXACTLY
    return contours, surfaces, np.array(circularity_array), np.array(circumferential_zero_vectors)


def main():
    # load the data
    data_directory = "David"

    centerline_diastole = pv.read(f"{data_directory}/Centerline_model_David.vtp")
    centerline_systole = pv.read(f"{data_directory}/Centerline warped.vtp")
    # tried to reverse the systole centerline points but for whatever reason it's actually fucking impossible
    full_surface_diastole = pv.read(f"{data_directory}/david_vdm.vtp")
    full_surface_systole = pv.read(f"{data_directory}/david_vdm_warped.vtp")
    # could also just warp by vector on diastole but oh well

    # Fit splines to the centerline points
    centerline_diastole_spline = fit_spline(centerline_diastole)
    centerline_systole_spline = fit_spline(centerline_systole, reverse=True)
    # add arc length to the spline
    centerline_diastole_spline = centerline_diastole_spline.compute_arc_length()
    centerline_systole_spline = centerline_systole_spline.compute_arc_length()

    centerline_displacement = centerline_systole_spline.points - centerline_diastole_spline.points
    centerline_diastole_spline.point_data.set_vectors(centerline_displacement, "Displacement")

    all_extracted_diastole, all_surfaces_diastole, circularity_diastole, circumferential_zero_vectors_diastole = (
        circularity_contours(
            full_surface_diastole,
            centerline_diastole_spline,
            clip_radius=50.0,
        )
    )
    all_extracted_systole, all_surfaces_systole, circularity_systole, circumferential_zero_vectors_systole = (
        circularity_contours(
            full_surface_systole,
            centerline_systole_spline,
            clip_radius=50.0,
        )
    )

    centerline_diastole_spline["Circularity"] = circularity_diastole
    centerline_systole_spline["Circularity"] = circularity_systole
    centerline_diastole_spline["zero_vector"] = circumferential_zero_vectors_diastole
    centerline_systole_spline["zero_vector"] = circumferential_zero_vectors_systole

    centerline_nearest_neighbor_mapping(centerline_diastole_spline, full_surface_diastole)
    full_surface_diastole = pyvista_surface_nearest_neighbor_mapping(full_surface_diastole, full_surface_systole)

    s = full_surface_diastole["arc_length"] / centerline_diastole_spline["arc_length"][-1]
    theta = (full_surface_diastole["Circumferential Coordinate"] + np.pi) / (2 * np.pi)
    full_surface_diastole.point_data.set_array(
        np.stack((s, theta), axis=1),
        name="surface_coordinates",
    )
    full_surface_diastole = full_surface_diastole.compute_derivative(scalars="surface_coordinates")
    s_grad = full_surface_diastole["gradient"][:, :3]
    theta_grad = full_surface_diastole["gradient"][:, 3:]
    full_surface_diastole.point_data.set_vectors(s_grad, name="s_gradient")
    full_surface_diastole.point_data.set_vectors(theta_grad, name="theta_gradient")

    centerline_diastole_spline.save(f"{data_directory}/David_spline_centerline_diastole.vtp")
    centerline_systole_spline.save(f"{data_directory}/David_spline_centerline_systole.vtp")

    all_extracted_diastole.save(f"{data_directory}/david_diastole_contours.vtp")
    all_surfaces_diastole.save(f"{data_directory}/david_diastole_cross_sections.vtp")

    all_extracted_systole.save(f"{data_directory}/david_systole_contours.vtp")
    all_surfaces_systole.save(f"{data_directory}/david_systole_cross_sections.vtp")

    full_surface_diastole.save(f"{data_directory}/david_diastole_surface.vtp")
    full_surface_systole.save("David/david_systole_surface.vtp")

    full_surface_diastole_warped = full_surface_diastole.warp_by_vector("Centerline Displacement")
    full_surface_diastole_warped = full_surface_diastole_warped.warp_by_vector("Nearest Surface Point Displacement")

    full_surface_diastole_warped = full_surface_diastole_warped.compute_derivative(scalars="surface_coordinates")
    s_grad = full_surface_diastole_warped["gradient"][:, :3]
    theta_grad = full_surface_diastole_warped["gradient"][:, 3:]
    full_surface_diastole_warped.point_data.set_vectors(s_grad, name="s_gradient")
    full_surface_diastole_warped.point_data.set_vectors(theta_grad, name="theta_gradient")

    full_surface_diastole_warped.save(f"{data_directory}/david_diastole_surface_warped.vtp")

    # NEXT: need a way to associate the surface nodes with the centerline nodes... nearest neighbor for now? Then check in paraview
    # NOTE: I kind of do this already in a different script... just find that and copy?

    if PLOT:
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

        plt.show()


if __name__ == "__main__":
    PLOT = False  # Set to True if you want to plot circularity shifting and the unwrapped surface
    main()
