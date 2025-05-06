import cheartio
import meshio
import numpy as np


def filter_by_percentile(data: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lower_percentile = np.percentile(data, lower)
    upper_percentile = np.percentile(data, upper)

    return data[(data >= lower_percentile) & (data <= upper_percentile)]


def weighted_least_squares(x_array: np.ndarray, y_array: np.ndarray, weight_array: np.ndarray) -> np.ndarray:
    """Return the weighted least squares solution for the system of equations x_array @ beta = y_array.

    Weights are prescribed by a one-dimensional array that is then diagonalized.

    Args:
        x_array (np.ndarray): _description_
        y_array (np.ndarray): _description_
        weight_array (np.ndarray): _description_

    Returns:
        np.ndarray: beta array

    """
    return np.linalg.inv(x_array.T @ np.diag(weight_array) @ x_array) @ x_array.T @ np.diag(weight_array) @ y_array


def calculate_area(points: np.ndarray, connectivity: np.ndarray) -> np.ndarray:
    """Find the areas of an array of triangles given their points and connectivity.

    Args:
        points (np.ndarray): ENTIRE X file (not a subset of it -- this will screw up the indexing)
        connectivity (np.ndarray): Boundary file with only the node indices

    Returns:
        np.ndarray: Nx1 array of areas of each triangle on boundary. Cell-centered calculation.

    """
    return 0.5 * np.linalg.norm(
        np.cross(
            points[connectivity[:, 1]] - points[connectivity[:, 0]],
            points[connectivity[:, 2]] - points[connectivity[:, 0]],
        ),
        axis=1,
    )


def construct_diastolic_matrix(
    central_node_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    xyz: np.ndarray,
    normal: np.ndarray,
):
    unique_neighbor_nodes_count = neighbor_indices.shape[0]
    diastolic_matrix = np.zeros((unique_neighbor_nodes_count * 3 + 3, 9))
    # actually this sucks... just focus on concatenation...

    for i, node_idx in enumerate(neighbor_indices):
        diastolic_submatrix = np.zeros((3, 9))
        displacement_from_central = xyz[node_idx] - xyz[central_node_idx]
        diastolic_submatrix[0, 0:3] = displacement_from_central
        diastolic_submatrix[1, 3:6] = displacement_from_central
        diastolic_submatrix[2, 6:9] = displacement_from_central

        # slot our submatrix into the right spot
        diastolic_matrix[i * 3 : (i + 1) * 3, :] = diastolic_submatrix.copy()

    diastolic_matrix[-3, 0:3] = normal
    diastolic_matrix[-2, 3:6] = normal
    diastolic_matrix[-1, 6:9] = normal

    return diastolic_matrix


def construct_systolic_vector(
    central_node_idx: np.ndarray,
    neighbor_indices: np.ndarray,
    xyz_systole: np.ndarray,
    normal: np.ndarray,
):
    unique_neighbor_nodes_count = neighbor_indices.shape[0]
    systolic_vector = np.zeros(unique_neighbor_nodes_count * 3 + 3)  # note that this is 2D

    for i, node_idx in enumerate(neighbor_indices):
        systolic_vector[i * 3 : (i + 1) * 3] = xyz_systole[node_idx] - xyz_systole[central_node_idx]

    systolic_vector[-3:] = normal

    return systolic_vector[:, np.newaxis]


def main():
    data_path = "will_data/varying circumferential stiffness/brandon_stuff_bulge"

    xyz, ien, _ = cheartio.read_mesh(f"{data_path}/cyl")

    cylindrical_coords0 = cheartio.read_dfile(f"{data_path}/Coord-0.D")
    cylindrical_coords100 = cheartio.read_dfile(f"{data_path}/Coord-100.D")

    unloaded_d = {
        "longitudinal_coord": cylindrical_coords0[:, 0:3],
        "circumferential_coord": cylindrical_coords0[:, 3:6],
        "radial_coord": cylindrical_coords0[:, 6:9],
        "space": cheartio.read_dfile(f"{data_path}/Space-0.D"),
        "stiffness": cheartio.read_dfile(f"{data_path}/Stiff-0.D"),
    }

    loaded_d = {
        "longitudinal_coord": cylindrical_coords100[:, 0:3],
        "circumferential_coord": cylindrical_coords100[:, 3:6],
        "radial_coord": cylindrical_coords100[:, 6:9],
        "space": cheartio.read_dfile(f"{data_path}/Space-100.D"),
        "stiffness": cheartio.read_dfile(f"{data_path}/Stiff-0.D"),
    }

    area_unloaded = calculate_area(unloaded_d["space"], ien)
    area_loaded = calculate_area(loaded_d["space"], ien)
    cell_strain = (area_loaded - area_unloaded) / area_unloaded

    # max Pressure is 10 kPa
    max_pressure = 10

    cells = [("triangle", ien)]

    # 1 means adjacent triangles, 2 means neighbors of neighbors, 3 means neighbors of neighbors of neighbors...
    for neighbor_extent in range(1, 5):
        circumferential_strain = np.zeros(xyz.shape[0])
        node_stiffness_wall = np.zeros(xyz.shape[0])
        node_stiffness_wall_proper = np.zeros(xyz.shape[0])
        systolic_circumferential_normals = np.zeros((xyz.shape[0], 3))

        # area stuff
        node_area_loaded_connected = np.zeros(xyz.shape[0])
        node_area_unloaded_connected = np.zeros(xyz.shape[0])
        node_element_count_connected = np.zeros(xyz.shape[0])

        for node_idx in np.unique(ien.flatten()):
            # find idx of all connected elements

            # FIND INITIAL CONNECTED TRIANGLES
            triangle_idx = np.argwhere(np.any(ien == node_idx, axis=1)).flatten()

            for _ in range(neighbor_extent - 1):
                connected_nodes = np.unique(ien[triangle_idx].flatten())
                connected_triangles = []
                for node in connected_nodes:
                    connected_triangles.extend(
                        np.argwhere(np.any(ien == node, axis=1)).flatten().tolist(),
                    )

                triangle_idx = np.unique(np.array(connected_triangles).flatten())

            triangle_count = len(triangle_idx)
            node_area_loaded_connected[node_idx] = np.sum(area_loaded[triangle_idx])
            node_area_unloaded_connected[node_idx] = np.sum(area_unloaded[triangle_idx])
            node_element_count_connected[node_idx] = triangle_count
            # NOW we have all triangles that we'd like to look at...
            unique_neighbor_node_idx = np.setdiff1d(np.unique(ien[triangle_idx]).flatten(), node_idx)

            diastolic_matrix = construct_diastolic_matrix(
                node_idx,
                unique_neighbor_node_idx,
                xyz,
                unloaded_d["radial_coord"][node_idx],
            )
            systolic_vector = construct_systolic_vector(
                node_idx,
                unique_neighbor_node_idx,
                loaded_d["space"],
                loaded_d["radial_coord"][node_idx],
            )

            # calculate the least squares solution
            # NOTE: starting with no weighting...
            deformation_gradient = weighted_least_squares(
                diastolic_matrix,
                systolic_vector,
                np.ones(diastolic_matrix.shape[0]),
            )

            deformation_gradient = deformation_gradient.reshape(3, 3, order="F")

            # FIXME: do I use the loaded circumferential or the unloaded circumferential?
            circumferential_strain[node_idx] = 0.5 * (
                np.linalg.norm(np.dot(deformation_gradient, unloaded_d["circumferential_coord"][node_idx])) ** 2 - 1
            )
            node_stiffness_wall_proper[node_idx] = max_pressure / circumferential_strain[node_idx]
            node_stiffness_wall[node_idx] = max_pressure / (circumferential_strain[node_idx] + 1)
            systolic_circumferential_normals[node_idx] = loaded_d["circumferential_coord"][node_idx]

        node_area_loaded_connected /= node_element_count_connected
        node_area_unloaded_connected /= node_element_count_connected

        node_strain_connected = (
            node_area_loaded_connected - node_area_unloaded_connected
        ) / node_area_unloaded_connected
        node_stiffness_connected = max_pressure / node_strain_connected

        meshio.write_points_cells(
            f"{data_path}/will_estimated_stiffness_k{neighbor_extent}.vtu",
            xyz,
            cells,
            point_data={
                "circumferential_strain": circumferential_strain,
                "area_strain": node_strain_connected,
                "node_stiffness_wall": node_stiffness_wall,
                "node_stiffness_wall_proper": node_stiffness_wall_proper,
                "systolic_circumferential_normals": systolic_circumferential_normals,
                "node_area_stiffness": node_stiffness_connected,
            },
            cell_data={"cell_strain": [cell_strain]},
        )


if __name__ == "__main__":
    main()
