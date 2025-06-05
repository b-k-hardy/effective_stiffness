# %%
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scienceplots
from scipy.interpolate import BSpline, make_splprep


def calc_circularity(area: float, perimeter: float) -> float:
    """Calculate the circularity of an aortic cross-section given its area and perimeter."""
    return 4 * np.pi * area / (perimeter**2)


# def distance_along_centerline = pv.filters.DistanceAlongLine()


def fit_spline(centerline: pv.PolyData, smoothness: float = 10, reverse: bool = False) -> pv.PolyData:
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

    centerline_diastole_spline.save("David/David_spline_centerline_diastole.vtp")
    centerline_systole_spline.save("David/David_spline_centerline_systole.vtp")

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

    all_extracted_diastole.save("David/david_diastole_contours.vtp")
    all_surfaces_diastole.save("David/david_diastole_cross_sections.vtp")

    all_extracted_systole.save("David/david_systole_contours.vtp")
    all_surfaces_systole.save("David/david_systole_cross_sections.vtp")

    # %%

    with plt.style.context(["science", "notebook"]):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        ax.set_title("Circularity of Aortic Cross-Sections")

        # problem: SPACING IS NOT UNIFORM...
        ax.plot(
            centerline_diastole_spline["arc_length"] / centerline_diastole_spline["arc_length"][-1],
            circularity_diastole,
            label=f"Diastole (length = {centerline_diastole_spline['arc_length'][-1]:.2f})",
        )
        ax.plot(
            centerline_systole_spline["arc_length"] / centerline_systole_spline["arc_length"][-1],
            circularity_systole,
            label=f"Systole (length = {centerline_systole_spline['arc_length'][-1]:.2f})",
        )
        ax.set_xlabel("Normalized Arc Length")
        ax.set_ylabel("Circularity")

        ax.legend()
        plt.show()


# %%
if __name__ == "__main__":
    main()
