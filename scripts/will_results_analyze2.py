import matplotlib.pyplot as plt
import meshio
import numpy as np
from scipy.stats import linregress


def main():
    bent_path = "brandon_stuff_bent"
    bulge_path = "brandon_stuff_bulge"
    neighbor_level = 2

    # LOAD DATA
    bent_data = meshio.read(f"{bent_path}/will_estimated_stiffness_k{neighbor_level}.vtu")
    bent_circumferential_stiffness = np.asarray(bent_data.point_data["node_stiffness_wall_proper"])
    bent_area_stiffness = np.asarray(bent_data.point_data["node_area_stiffness"])

    bulge_data = meshio.read(f"{bulge_path}/will_estimated_stiffness_k{neighbor_level}.vtu")
    bulge_circumferential_stiffness = np.asarray(bulge_data.point_data["node_stiffness_wall_proper"])
    bulge_area_stiffness = np.asarray(bulge_data.point_data["node_area_stiffness"])

    # GROUND TRUTH
    bent_gt = meshio.read(f"{bent_path}/cyl0.vtu")
    bent_gt_stiffness = np.asarray(bent_gt.point_data["stiffness"])

    bulge_gt = meshio.read(f"{bulge_path}/cyl0.vtu")
    bulge_gt_stiffness = np.asarray(bulge_gt.point_data["stiffness"])
    # bulge_gt_stiffness = bulge_gt_stiffness[:, 0]  # NOTE: why the fuck is there an extra column lol

    # LINEAR REGRESSIONS
    lin_data = linregress(bent_gt_stiffness, bent_circumferential_stiffness)
    bent_circ_line = lin_data.slope * np.array([29, 61]) + lin_data.intercept

    lin_data2 = linregress(bent_gt_stiffness, bent_area_stiffness)
    bent_area_line = lin_data2.slope * np.array([29, 61]) + lin_data2.intercept

    # PLOT FAT
    fig, ax = plt.subplots(1, 2, layout="constrained", sharey=True, sharex=True, figsize=(7, 2.5))
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[0].text(
        0.05,
        0.60,
        f"slope = {np.round(lin_data.slope, 3)}\nintercept = {np.round(lin_data.intercept, 3)}\n$r^2$ = {np.round(lin_data.rvalue**2, 3)}",
        transform=ax[0].transAxes,
    )
    ax[0].scatter(bent_gt_stiffness, bent_circumferential_stiffness)
    ax[0].plot([29, 61], bent_circ_line, color="k", linestyle="--")
    ax[0].set(
        title="Circumferential Strain-based Stiffness",
        xlabel="Ground Truth Stiffness",
        ylabel="Estimated Stiffness",
        ylim=(5, 20),
    )
    ax[1].text(
        0.55,
        0.10,
        f"slope = {np.round(lin_data2.slope, 3)}\nintercept = {np.round(lin_data2.intercept, 3)}\n$r^2$ = {np.round(lin_data2.rvalue**2, 3)}",
        transform=ax[1].transAxes,
    )
    ax[1].scatter(bent_gt_stiffness, bent_area_stiffness, color="tab:orange")
    ax[1].plot([29, 61], bent_area_line, color="k", linestyle="--")
    ax[1].set(title="Area Strain-based Stiffness", xlabel="Ground Truth Stiffness", ylim=(5, 20))
    fig.suptitle("Varying Circumferential Stiffness -- Bent")
    fig.savefig("vary_circumferential_bent_stiffness.pdf")

    # LINEAR REGRESSIONS
    lin_data3 = linregress(bulge_gt_stiffness, bulge_circumferential_stiffness)
    bulge_circ_line = lin_data3.slope * np.array([29, 61]) + lin_data3.intercept

    lin_data4 = linregress(bulge_gt_stiffness, bulge_area_stiffness)
    bulge_area_line = lin_data4.slope * np.array([29, 61]) + lin_data4.intercept

    # PLOT SKINNY
    fig, ax = plt.subplots(1, 2, layout="constrained", sharey=True, sharex=True, figsize=(7, 2.5))
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")
    ax[0].text(
        0.05,
        0.60,
        f"slope = {np.round(lin_data3.slope, 3)}\nintercept = {np.round(lin_data3.intercept, 3)}\n$r^2$ = {np.round(lin_data3.rvalue**2, 3)}",
        transform=ax[0].transAxes,
    )
    ax[0].plot([29, 61], bulge_circ_line, color="k", linestyle="--")
    ax[0].scatter(bulge_gt_stiffness, bulge_circumferential_stiffness)
    ax[0].set(
        title="Circumferential Strain-based Stiffness",
        xlabel="Ground Truth Stiffness",
        ylabel="Estimated Stiffness",
        ylim=(5, 20),
    )
    ax[1].text(
        0.20,
        0.60,
        f"slope = {np.round(lin_data4.slope, 3)}\nintercept = {np.round(lin_data4.intercept, 3)}\n$r^2$ = {np.round(lin_data4.rvalue**2, 3)}",
        transform=ax[1].transAxes,
    )
    ax[1].plot([29, 61], bulge_area_line, color="k", linestyle="--")
    ax[1].scatter(bulge_gt_stiffness, bulge_area_stiffness, color="tab:orange")
    ax[1].set(title="Area Strain-based Stiffness", xlabel="Ground Truth Stiffness", ylim=(5, 20))
    fig.suptitle("Varying Circumferential Stiffness -- Bulge")
    fig.savefig("vary_circumferential_bulge_stiffness.pdf")


if __name__ == "__main__":
    main()
