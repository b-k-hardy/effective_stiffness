# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scienceplots
import vtkmodules.util.vtkConstants as vtk
from scipy.interpolate import make_splprep
from scipy.spatial import KDTree
from vtkmodules import vtkCommonCore
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkStaticCellLocator


def pyvista_surface_nearest_neighbor_mapping(mesh_d: pv.PolyData, mesh_s: pv.PolyData) -> pv.PolyData:
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement")
    closest_cells, closest_points = mesh_s.find_closest_cell(mesh_d.points, return_closest_point=True)
    displacements = closest_points - mesh_d.points
    mesh_d.point_data.set_vectors(
        displacements,
        "Distance to Nearest Implicit Surface Point PYVISTA",
    )
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement", factor=-1.0)  # Reverse the warping direction

    return mesh_d


# USELESS FUNCTION NOW
def new_surface_nearest_neighbor_mapping(mesh_d: pv.PolyData, mesh_s: pv.PolyData) -> pv.PolyData:
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement")
    # vtk_mesh = mesh_d.GetDataObject()  # Get the underlying VTK mesh from the PyVista PolyData
    # not sure if I have to do this...

    # Step 2: Create a spatial locator for the mesh
    # This builds an acceleration structure (like a BVH) for fast closest-point queries
    locator = vtkStaticCellLocator()
    locator.SetDataSet(mesh_s)  # Assign the mesh
    locator.BuildLocator()  # Preprocess for fast spatial queries

    # Step 3: Define the point you're querying from
    points = mesh_d.points  # Points from the source mesh (e.g., systole)

    displacements = []
    for point in points:
        # Step 4: Prepare VTK-compatible variables to receive results
        closest_point = [0.0, 0.0, 0.0]  # Will hold closest point coordinates
        cell_id = vtkCommonCore.reference(0)  # Will hold the ID of the cell found
        sub_id = vtkCommonCore.reference(0)  # Will hold sub-ID inside the cell (e.g., triangle index)
        dist2 = vtkCommonCore.reference(0.0)  # Will hold squared distance result

        # Step 5: Query the locator for the closest point on the mesh to the given point
        locator.FindClosestPoint(
            point,  # The input query point
            closest_point,  # Output: closest point on the mesh
            cell_id,  # Output: cell ID (not used here, but required)
            sub_id,  # Output: sub ID
            dist2,  # Output: squared distance (not used directly here)
        )

        # Step 6: Convert result to NumPy array for easy math
        closest_point_np = np.array(closest_point)

        # Step 7: Compute the vector from the original point to the closest point
        displacements.append(closest_point_np - point)

    displacements = np.array(displacements)
    mesh_d.point_data.set_vectors(
        displacements,
        "Distance to Nearest Implicit Surface Point",
    )
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement", factor=-1.0)  # Reverse the warping direction

    return mesh_d


# wrong distance
def surface_nearest_neighbor_mapping(mesh_d: pv.PolyData, mesh_s: pv.PolyData) -> pv.PolyData:
    """Find the nearest neighbor indices in mesh_s for each point in mesh_d.

    Args:
        mesh_d (pv.PolyData): Source mesh (e.g., diastole).
        mesh_s (pv.PolyData): Target mesh (e.g., systole).

    Returns:
        np.ndarray: Array of indices in mesh_s corresponding to nearest neighbors for each point in mesh_d.

    """
    mesh_d = mesh_d.warp_by_vector("Centerline Displacement")
    tree = KDTree(mesh_d.points)
    distances, indices = tree.query(mesh_s.points)
    mesh_d.point_data.set_vectors(
        mesh_s.points[indices] - mesh_d.points,
        "Distance to Nearest Neighbor",
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
        centerline_displacement[i] = centerline_spline["Displacement"][centerline_idx]
        additional_displacement[i] = mesh["Displacement"][i] - centerline_spline["Displacement"][centerline_idx]

    # assign the circularity and arc length values to the mesh point data
    mesh.point_data.set_array(mesh_circularity, "Circularity")
    mesh.point_data.set_array(mesh_arc_length, "arc_length")
    mesh.point_data.set_array(centerline_displacement, "Centerline Displacement")
    mesh.point_data.set_array(additional_displacement, "Additional Displacement")


def calc_circularity(area: float, perimeter: float) -> float:
    """Calculate the circularity of an aortic cross-section given its area and perimeter."""
    return 4 * np.pi * area / (perimeter**2)


# def distance_along_centerline = pv.filters.DistanceAlongLine()


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
) -> pv.PolyData:
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


# %%
def main():
    # %%
    # load the data
    centerline_diastole = pv.read("David/Centerline_model_David.vtp")
    centerline_systole = pv.read("David/Centerline warped.vtp")
    # tried to reverse the systole centerline points but for whatever reason it's actually fucking impossible
    full_surface_diastole = pv.read("David/david_vdm.vtp")
    full_surface_systole = pv.read(
        "David/david_vdm_warped.vtp",
    )  # could also just warp by vector on diastole but oh well

    # Fit splines to the centerline points
    centerline_diastole_spline = fit_spline(centerline_diastole)
    centerline_systole_spline = fit_spline(centerline_systole, reverse=True)
    # add arc length to the spline
    centerline_diastole_spline = centerline_diastole_spline.compute_arc_length()
    centerline_systole_spline = centerline_systole_spline.compute_arc_length()

    centerline_displacement = centerline_systole_spline.points - centerline_diastole_spline.points
    centerline_diastole_spline.point_data.set_vectors(centerline_displacement, "Displacement")

    all_extracted_diastole, all_surfaces_diastole, circularity_diastole = circularity_contours(
        full_surface_diastole,
        centerline_diastole_spline,
        clip_radius=35.0,
    )
    all_extracted_systole, all_surfaces_systole, circularity_systole = circularity_contours(
        full_surface_systole,
        centerline_systole_spline,
        clip_radius=38.0,
    )

    centerline_diastole_spline.point_data.set_scalars(circularity_diastole, "Circularity")
    centerline_systole_spline.point_data.set_scalars(circularity_systole, "Circularity")

    centerline_nearest_neighbor_mapping(centerline_diastole_spline, full_surface_diastole)

    # tests to remove
    full_surface_diastole = surface_nearest_neighbor_mapping(full_surface_diastole, full_surface_systole)
    full_surface_diastole = new_surface_nearest_neighbor_mapping(full_surface_diastole, full_surface_systole)
    # keep
    full_surface_diastole = pyvista_surface_nearest_neighbor_mapping(full_surface_diastole, full_surface_systole)
    # centerline_nearest_neighbor_mapping(centerline_systole_spline, full_surface_systole)

    centerline_diastole_spline.save("David/David_spline_centerline_diastole.vtp")
    centerline_systole_spline.save("David/David_spline_centerline_systole.vtp")

    all_extracted_diastole.save("David/david_diastole_contours.vtp")
    all_surfaces_diastole.save("David/david_diastole_cross_sections.vtp")

    all_extracted_systole.save("David/david_systole_contours.vtp")
    all_surfaces_systole.save("David/david_systole_cross_sections.vtp")

    full_surface_diastole.save("David/david_diastole_surface.vtp")
    # full_surface_systole.save("David/david_systole_surface.vtp")

    # NEXT: need a way to associate the surface nodes with the centerline nodes... nearest neighbor for now? Then check in paraview
    # NOTE: I kind of do this already in a different script... just find that and copy?

    # %%

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
        plt.show()


# %%
if __name__ == "__main__":
    main()
