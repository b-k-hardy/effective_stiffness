import cheartio
import matplotlib.pyplot as plt
import meshio
import numpy as np

K = 1e4  # stiffness for spatial multiplier map. (Pa/mm)


def calculate_area(points: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Find the areas of an array of triangles given their points and connectivity.

    Args:
        points (np.ndarray): _description_
        connectivity (np.ndarray): _description_

    Returns:
        np.ndarray: _description_

    """
    return 0.5 * np.linalg.norm(
        np.cross(
            points[connectivity[:, 1]] - points[connectivity[:, 0]],
            points[connectivity[:, 2]] - points[connectivity[:, 0]],
        ),
        axis=1,
    )


def displacement_stiffness(xyz, bfile, systole_d, diastole_d):
    return -1


def area_stiffness():
    return -1


def load_data(path, systole_timestep):
    xyz, _, _ = cheartio.read_mesh(path)
    bfile = cheartio.read_bfile(path)

    # NOTE: d to vtu writer: need to input list of D files and then loop through timesteps.
    # path_t0 = ["X.D", "P.D"] etc.

    # load in systole and diastole data and set up data dictionaries
    systole_d = {
        "X": cheartio.read_dfile(f"X-{systole_timestep}.D"),
        "P": cheartio.read_dfile(f"P-{systole_timestep}.D"),
        "spatK": cheartio.read_dfile(f"spatK-{systole_timestep}.D"),
    }
    diastole_d = {
        "X": cheartio.read_dfile("X-1.D"),
        "P": cheartio.read_dfile("P-1.D"),
        "spatK": cheartio.read_dfile("spatK-1.D"),
    }

    return xyz, bfile, systole_d, diastole_d


def main():
    # read in geometry
    xyz, bfile, systole_d, diastole_d = load_data("model", 241)
    node_normals = cheartio.read_dfile("norms.D")


    ################## Calculate Displacement Stiffness ##################
    # calculate displacement field
    displacement = systole_d["X"] - diastole_d["X"]

    # find all nodes that exist on outer wall; flatten and take unique values
    wall_nodes = np.unique(bfile[np.isin(bfile[:, 4], 8), 1:-1].flatten())

    # find nodes that have moved at all (1e-7 tolerance)
    moving_nodes = np.abs(np.linalg.norm(displacement, axis=1)) > 1e-7
    # instead of having a giant true/false array, we are going to grab the row indices of the moving nodes
    moving_nodes_idx = np.arange(displacement.shape[0])[moving_nodes]

    # find nodes on the outer wall boundary that have also moved (pretty much all except for boundary edges
    moving_wall_nodes = np.intersect1d(wall_nodes, moving_nodes_idx)

    displacement_outer_wall = displacement[moving_wall_nodes]

    node_normals = node_normals[moving_wall_nodes]

    pressure_delta = systole_d["P"] - diastole_d["P"]
    pressure_delta_outer_wall = pressure_delta[moving_wall_nodes]

    # Math: k = \frac{(P_{\text{sys}}-P_{\text{dia}})}{\left\lVert u_{\text{sys}}\right\rVert^2_2} u_{\text{sys}}^T n
    # Calculate spring constant from displacement and pressure data
    k = (
        pressure_delta_outer_wall
        / np.linalg.norm(displacement_outer_wall, axis=1) ** 2
        * np.sum(displacement_outer_wall * node_normals, axis=1)
    )

    # THEREFORE any point data will need to maintain original indexing... oh fuck so I need a lot of zeros lol
    stiffness_points = np.zeros(np.shape(xyz)[0])
    stiffness_points[moving_wall_nodes] = k
    stiffness_points[stiffness_points == 0] = np.nan

    print("Mean K:", np.mean(k))
    print("Std Dev K:", np.std(k))
    print("Max K:", np.max(k))
    print("Min K:", np.min(k))

    ################## Calculate Area Stiffness ##################
    wall_bfile_elements = bfile[np.isin(bfile[:, 4], 8), 1:-1]  # <- this has original indexing from X file...
    area_sys = calculate_area(systole_d["X"], wall_bfile_elements)
    area_dia = calculate_area(diastole_d["X"], wall_bfile_elements)
    strain = (area_sys - area_dia) / area_dia

    # just see if I can add areas to each node, then average by # of adjacent cells, then find node strain.
    # MAYBE multiply avg by 2 because that's what the expression that david gave me evaluates to for a 6 neighbor node

    node_area_sys = np.zeros(xyz.shape[0])
    node_area_dia = np.zeros(xyz.shape[0])
    # this keeps track of how many elements have a certain node in their connectivity.
    # This is so we can calculate area averages for each node.
    node_element_count = np.zeros(xyz.shape[0])

    for i in range(len(wall_bfile_elements.flatten())):
        node_idx = wall_bfile_elements.flatten()[i]
        node_area_sys[node_idx] += area_sys[i // 3]
        node_area_dia[node_idx] += area_dia[i // 3]
        node_element_count[node_idx] += 1

    node_area_sys /= 0.5 * node_element_count
    node_area_dia /= 0.5 * node_element_count

    node_strain = (node_area_sys - node_area_dia) / node_area_dia
    node_stiffness = pressure_delta / node_strain
    node_stiffness_wall = pressure_delta_outer_wall / node_strain[moving_wall_nodes]

    # FIXME: I'm definitely fucking up the indexing BIG TIME...
    # node_strain[wall_bfile_elements] = strain[wall_bfile_elements]

    # node_strain[3331]  # WAIT shouldn't I add... some indices are repeated here right?

    np.sum(node_strain)

    # node_stiffness = pressure_delta
    # YES or wait... should I assemble area THEN calculate strain? don't thnk it matters

    ################ Write results to vtu files ################
    cells = [("triangle", wall_bfile_elements)]

    meshio.write_points_cells(
        "estimated_stiffness.vtu",
        xyz,
        cells,
        point_data={
            "Stiffness (Pa/mm)": stiffness_points,
            "Node Strain": node_strain,
            "Node Stiffness": node_stiffness,
        },
        cell_data={"Strain": [strain]},
    )
    meshio.write_points_cells(
        "prescribed_stiffness.vtu",
        xyz,
        cells,
        point_data={"Stiffness (Pa/mm)": K * systole_d["spatK"]},
    )
    meshio.write_points_cells(
        "stiffness_error.vtu",
        xyz,
        cells,
        point_data={
            "Absolute Error (Pa/mm)": np.abs(K * systole_d["spatK"] - stiffness_points),
            "Relative Error": np.abs(K * systole_d["spatK"] - stiffness_points) / (K * systole_d["spatK"]),
        },
    )

    # could do a bunch of violin plots here to show the distribution of k values depending on the region...
    plt.violinplot(node_stiffness_wall, showextrema=False, quantiles=[0.25, 0.5, 0.75])
    plt.title("Distribution of K values on walls")
    plt.xlabel("Method")

    plt.show()


if __name__ == "__main__":
    main()
