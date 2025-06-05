import numpy as np
import pyvista as pv
from scipy.interpolate import splev, splprep


def circularity(area: float, perimeter: float) -> float:
    """Calculate the circularity of an aortic cross-section given its area and perimeter."""
    return 4 * np.pi * area / (perimeter**2)


def fit_spline(centerline: pv.PolyData) -> pv.PolyData:
    """Fit a spline to the centerline points."""
    tck, u = splprep([centerline.points[:, 0], centerline.points[:, 1], centerline.points[:, 2]])
    new_points = splev(u, tck)
    first_deriv = splev(u, tck, der=1)

    new_points = np.array(new_points).T
    first_deriv = np.array(first_deriv).T

    return new_points, first_deriv


def main():
    # load the centerline data
    centerline_diastole = pv.read("Centerline_model_David.vtp")
    centerline_systole = pv.read("Centerline warped.vtp")

    # Fit splines to the centerline points
    centerline_diastole_spline = fit_spline(centerline_diastole)
    centerline_warped_points = centerline_systole.points

    tck, u = splprep([main_centerline[:, 0], main_centerline[:, 1], main_centerline[:, 2]])
    new_points = splev(u, tck)
    first_deriv = splev(u, tck, der=1)

    new_points = np.array(new_points).T
    first_deriv = np.array(first_deriv).T

    # Create a PolyData object from new_points
    new_points_polydata = pv.PolyData(new_points, lines=[358] + np.arange(len(new_points)).tolist())

    # Save the PolyData object to a .vtp file
    new_points_polydata.point_data.set_vectors(first_deriv, "Derivative")

    new_points_polydata.save("new_points.vtp")

    tck, u = splprep([centerline_warped_points[:, 0], centerline_warped_points[:, 1], centerline_warped_points[:, 2]])
    new_points_warped = splev(u, tck)
    first_deriv_warped = splev(u, tck, der=1)

    new_points_warped = np.array(new_points_warped).T
    first_deriv_warped = np.array(first_deriv_warped).T

    # Create a PolyData object from new_points
    new_points_polydata_warped = pv.PolyData(
        new_points_warped, lines=[344] + np.arange(len(new_points_warped)).tolist()
    )

    # Save the PolyData object to a .vtp file
    new_points_polydata_warped.point_data.set_vectors(first_deriv_warped, "Derivative")

    new_points_polydata_warped.save("new_points_warped.vtp")

    full_model = pv.read("david_vdm.vtp")

    extracted_cuts = []
    extracted_surfaces = []

    for i in range(len(new_points)):
        roi = pv.Sphere(center=new_points[i], radius=35)

        test_slice = full_model.slice(origin=new_points[i], normal=first_deriv[i])
        test_slice.clear_data()

        extracted = test_slice.clip_surface(roi, invert=True)
        extracted_surf = extracted.delaunay_2d()

        lengths = extracted.compute_cell_sizes()

        perimeter = np.sum(lengths["Length"])
        extracted.cell_data.set_scalars(np.ones(extracted.n_cells) * perimeter, "Perimeter")
        extracted.cell_data.set_scalars(
            np.ones(extracted.n_cells) * circularity(extracted_surf.area, perimeter), "Circularity"
        )

        extracted_cuts.append(extracted)
        extracted_surfaces.append(extracted_surf)

    all_extracted = pv.merge(extracted_cuts)
    all_surfaces = pv.merge(extracted_surfaces)

    all_extracted.save("all_extracted.vtp")
    all_surfaces.save("all_surfaces.vtp")

    full_model = pv.read("david_vdm_warped.vtp")
    extracted_cuts_warped = []
    extracted_surfaces_warped = []

    for i in range(len(new_points_warped)):
        roi = pv.Sphere(center=new_points_warped[i], radius=35)

        test_slice = full_model.slice(origin=new_points_warped[i], normal=first_deriv_warped[i])
        test_slice.clear_data()

        extracted = test_slice.clip_surface(roi, invert=True)
        extracted_surf = extracted.delaunay_2d()

        lengths = extracted.compute_cell_sizes()

        perimeter = np.sum(lengths["Length"])
        extracted.cell_data.set_scalars(np.ones(extracted.n_cells) * perimeter, "Perimeter")
        extracted.cell_data.set_scalars(
            np.ones(extracted.n_cells) * circularity(extracted_surf.area, perimeter), "Circularity"
        )

        extracted_cuts_warped.append(extracted)
        extracted_surfaces_warped.append(extracted_surf)

    all_extracted = pv.merge(extracted_cuts_warped)
    all_surfaces = pv.merge(extracted_surfaces_warped)

    all_extracted.save("all_extracted_warped.vtp")
    all_surfaces.save("all_surfaces_warped.vtp")


if __name__ == "__main__":
    main()
