import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.stats import linregress


def main():
    gt_fat_path = "brandon_stuff_bent 2"
    gt_skinny_path = "brandon_stuff_bent"

    # LOAD DATA
    fat_data = meshio.read("will_estimated_stiffness_fat_k2.vtu")
    fat_circumferential_stiffness = np.asarray(fat_data.point_data["node_stiffness_wall_proper"])
    fat_area_stiffness = np.asarray(fat_data.point_data["node_area_stiffness"])

    skinny_data = meshio.read("will_estimated_stiffness_k2.vtu")
    skinny_circumferential_stiffness = np.asarray(skinny_data.point_data["node_stiffness_wall_proper"])
    skinny_area_stiffness = np.asarray(skinny_data.point_data["node_area_stiffness"])

    # GROUND TRUTH
    fat_gt = meshio.read(f"{gt_fat_path}/cyl0.vtu")
    fat_gt_stiffness = np.asarray(fat_gt.point_data["stiffness"])

    skinny_gt = meshio.read(f"{gt_skinny_path}/cyl0.vtu")
    skinny_gt_stiffness = np.asarray(skinny_gt.point_data["stiffness"])
    skinny_gt_stiffness = skinny_gt_stiffness[:, 0]  # NOTE: why the fuck is there an extra column lol

    # LINEAR REGRESSIONS
    lin_data = linregress(fat_gt_stiffness, fat_circumferential_stiffness)
    fat_circ_line = lin_data.slope * np.array([29, 61]) + lin_data.intercept

    lin_data2 = linregress(fat_gt_stiffness, fat_area_stiffness)
    fat_area_line = lin_data2.slope * np.array([29, 61]) + lin_data2.intercept

    # PLOT FAT
    fig, ax = plt.subplots(1, 2, layout="constrained", sharey=True, sharex=True, figsize=(7, 2.75))
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[0].text(
        0.05,
        0.75,
        f"slope = {np.round(lin_data.slope, 3)}\nintercept = {np.round(lin_data.intercept, 3)}\n$r^2$ = {np.round(lin_data.rvalue**2, 3)}",
        transform=ax[0].transAxes,
    )
    ax[0].scatter(fat_gt_stiffness, fat_circumferential_stiffness)
    ax[0].plot([29, 61], fat_circ_line, color="k", linestyle="--")
    ax[0].set(
        title="Circumferential Strain-based Stiffness",
        xlabel="Ground Truth Stiffness",
        ylabel="Estimated Stiffness",
    )
    ax[1].text(
        0.05,
        0.75,
        f"slope = {np.round(lin_data2.slope, 3)}\nintercept = {np.round(lin_data2.intercept, 3)}\n$r^2$ = {np.round(lin_data2.rvalue**2, 3)}",
        transform=ax[1].transAxes,
    )
    ax[1].scatter(fat_gt_stiffness, fat_area_stiffness, color="tab:orange")
    ax[1].plot([29, 61], fat_area_line, color="k", linestyle="--")
    ax[1].set(title="Area Strain-based Stiffness", xlabel="Ground Truth Stiffness")
    fig.suptitle("Large Diameter to Radius of Curvature Ratio")
    fig.savefig("fat_stiffness.svg")

    # LINEAR REGRESSIONS
    lin_data3 = linregress(skinny_gt_stiffness, skinny_circumferential_stiffness)
    skinny_circ_line = lin_data3.slope * np.array([29, 61]) + lin_data3.intercept

    lin_data4 = linregress(skinny_gt_stiffness, skinny_area_stiffness)
    skinny_area_line = lin_data4.slope * np.array([29, 61]) + lin_data4.intercept

    # PLOT SKINNY
    fig, ax = plt.subplots(1, 2, layout="constrained", sharey=True, sharex=True, figsize=(7, 3.5))
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[0].text(
        0.05,
        0.80,
        f"slope = {np.round(lin_data3.slope, 3)}\nintercept = {np.round(lin_data3.intercept, 3)}\n$r^2$ = {np.round(lin_data3.rvalue**2, 3)}",
        transform=ax[0].transAxes,
    )
    ax[0].plot([29, 61], skinny_circ_line, color="k", linestyle="--")
    ax[0].scatter(skinny_gt_stiffness, skinny_circumferential_stiffness)
    ax[0].set(
        title="Circumferential Strain-based Stiffness",
        xlabel="Ground Truth Stiffness",
        ylabel="Estimated Stiffness",
    )
    ax[1].text(
        0.05,
        0.80,
        f"slope = {np.round(lin_data4.slope, 3)}\nintercept = {np.round(lin_data4.intercept, 3)}\n$r^2$ = {np.round(lin_data4.rvalue**2, 3)}",
        transform=ax[1].transAxes,
    )
    ax[1].plot([29, 61], skinny_area_line, color="k", linestyle="--")
    ax[1].scatter(skinny_gt_stiffness, skinny_area_stiffness, color="tab:orange")
    ax[1].set(title="Area Strain-based Stiffness", xlabel="Ground Truth Stiffness")
    fig.suptitle("Small Diameter to Radius of Curvature Ratio")
    fig.savefig("skinny_stiffness.svg")


if __name__ == "__main__":
    main()
