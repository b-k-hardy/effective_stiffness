import matplotlib.pyplot as plt
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

    for i in range(centerline_sp.n_points):
        roi = pv.Sphere(center=centerline_sp.points[i], radius=clip_radius)

        test_slice = full_surface.slice(origin=centerline_sp.points[i], normal=centerline_sp["Derivative"][i])
        test_slice.clear_data()

        extracted = test_slice.clip_surface(roi, invert=True)
        extracted_surf = extracted.delaunay_2d()

        lengths = extracted.compute_cell_sizes()

        perimeter = np.sum(lengths["Length"])
        extracted.cell_data.set_scalars(np.ones(extracted.n_cells) * perimeter, "Perimeter")
        extracted.cell_data.set_scalars(
            np.ones(extracted.n_cells) * circularity(extracted_surf.area, perimeter),
            "Circularity",
        )

        extracted_cuts.append(extracted)
        extracted_surfaces.append(extracted_surf)

    contours = pv.merge(extracted_cuts)
    surfaces = pv.merge(extracted_surfaces)

    return contours, surfaces


def main():
    # load the data
    centerline_diastole = pv.read("David/Centerline_model_David.vtp")
    centerline_systole = pv.read("David/Centerline warped.vtp")
    full_surface_diastole = pv.read("David/david_vdm.vtp")
    full_surface_systole = pv.read(
        "David/david_vdm_warped.vtp",
    )  # could also just warp by vector on diastole but oh well

    # Fit splines to the centerline points
    centerline_diastole_spline = fit_spline(centerline_diastole)
    centerline_systole_spline = fit_spline(centerline_systole)

    centerline_diastole_spline.save("David/David_spline_centerline_diastole.vtp")
    centerline_systole_spline.save("David/David_spline_centerline_systole.vtp")

    all_extracted_diastole, all_surfaces_diastole = circularity_contours(
        full_surface_diastole,
        centerline_diastole_spline,
        clip_radius=35.0,
    )
    all_extracted_systole, all_surfaces_systole = circularity_contours(
        full_surface_systole,
        centerline_systole_spline,
        clip_radius=38.0,
    )

    all_extracted_diastole.save("David/david_diastole_contours.vtp")
    all_surfaces_diastole.save("David/david_diastole_cross_sections.vtp")

    all_extracted_systole.save("David/david_systole_contours.vtp")
    all_surfaces_systole.save("David/david_systole_cross_sections.vtp")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    ax[0].set_title("Diastole Contours")
    ax[1].set_title("Systole Contours")

    ax[0].plot(all_extracted_diastole["Circularity"])
    ax[1].plot(all_extracted_systole["Circularity"])

    plt.show()


if __name__ == "__main__":
    main()
