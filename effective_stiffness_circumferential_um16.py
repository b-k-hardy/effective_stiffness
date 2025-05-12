import numpy as np
import pyvista as pv


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


def estimate_normal(
    central_node_idx: np.ndarray,
    diastolic_normals: np.ndarray,
    neighbor_triangle_indices: np.ndarray,
    xyz: np.ndarray,
):
    # take in the relevent triangle indices and calculate the normal
    # need to check orientation of each cell normal with the norms.D file that Mia gave me
    # this is not something I'll always have, but it's a good place to start :D
    # then find average normal

    normals = np.zeros((neighbor_triangle_indices.shape[0], 3))

    for i in range(neighbor_triangle_indices.shape[0]):
        # calculate normal
        # check orientation
        # if correct, return normal
        # else, return -normal
        point1 = xyz[neighbor_triangle_indices[i, 0]]
        point2 = xyz[neighbor_triangle_indices[i, 1]]
        point3 = xyz[neighbor_triangle_indices[i, 2]]
        vec1 = point2 - point1
        vec2 = point3 - point1
        normal = np.cross(vec1, vec2)

        if np.dot(normal, diastolic_normals[central_node_idx]) > 0:
            normal = normal / np.linalg.norm(normal)
        else:
            normal = -normal / np.linalg.norm(normal)
        normals[i] = normal

    avg_normal = np.mean(normals, axis=0)

    return avg_normal / np.linalg.norm(avg_normal)


def find_circumferential_direction(
    centerline_points: np.ndarray,
    centerline_normals: np.ndarray,
    surface_normal: np.ndarray,
    node_xyz: np.ndarray,
):
    # find the closest point on the centerline
    # find the normal at that point
    # find the circumferential direction
    # return the circumferential direction
    centerline_idx = np.argmin(np.linalg.norm(centerline_points - node_xyz, axis=1))
    centerline_normal = centerline_normals[centerline_idx]

    circfumferential_direction = np.cross(centerline_normal, surface_normal)

    return circfumferential_direction / np.linalg.norm(circfumferential_direction)


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

    # FIXME: need to add systolic normal here.
    # THIS WON'T WORK BECAUSE I DON'T HAVE THE TRIANGLES HERE...
    systolic_vector[-3:] = normal

    return systolic_vector[:, np.newaxis]


def main():
    # load in the data!
    # centerline data
    source_directory = "UM16"
    smoothed_centerline = pv.read(f"{source_directory}/new_points.vtp")
    centerline_points = smoothed_centerline.points
    centerline_normals = smoothed_centerline.point_data["Derivative"]

    # vdm and mesh data
    vdm = pv.read(f"{source_directory}/UM16_VDM-D.vtp")
    diastole_nodes = vdm.points
    diastole_normals = vdm.point_data["SurfaceNormals"]  # I really don't know if this is the diastolic normal or not
    displacement = vdm.point_data["Displacement"]
    systole_nodes = diastole_nodes + displacement

    # NOTE: TEST to see if this gives normals:
    vdm_systole = vdm.warp_by_vector("Displacement")

    systole_normals = vdm_systole.compute_normals()
    systole_normals = systole_normals.point_data["Normals"]

    # pressure data
    # FIXME: need to do the paraview conversion!

    ###############################################################################
    # read in geometry and data

    # NOTE: watch out for tiny displacements... this could be a problem!

    # 1 means adjacent triangles, 2 means neighbors of neighbors, 3 means neighbors of neighbors of neighbors...
    for neighbor_extent in range(1, 5):
        def_gradient_determinant = np.zeros(diastole_nodes.shape[0])
        circumferential_strain = np.zeros(diastole_nodes.shape[0])
        node_stiffness_wall = np.zeros(diastole_nodes.shape[0])
        node_stiffness_wall_proper = np.zeros(diastole_nodes.shape[0])
        systolic_node_normals = np.zeros((diastole_nodes.shape[0], 3))
        systolic_circumferential_normals = np.zeros((diastole_nodes.shape[0], 3))

        # shit this index is actually important lol...
        for node_idx in range(vdm.points.shape[0]):
            # find idx of all connected elements
            neighbor_points_idx = vdm.point_neighbors_levels(node_idx, neighbor_extent)
            neighbor_points_idx = list(neighbor_points_idx)
            neighbor_points_idx = [item for sublist in neighbor_points_idx for item in sublist]
            neighbor_points_idx = np.array(neighbor_points_idx)

            # NOTE: the code above actually distinguishes between the neighbor level away... could maybe use this thing OUTSIDE of the loop...

            # Oh fuck I actually need the adjacent triangles here...
            # systolic_node_normal = estimate_normal(
            #    node_idx,
            #    diastole_normals,
            #    neighbor_points_idx,
            #    systole_nodes,
            # )

            diastolic_matrix = construct_diastolic_matrix(
                node_idx,
                neighbor_points_idx,
                diastole_nodes,
                diastole_normals[node_idx],
            )
            systolic_vector = construct_systolic_vector(
                node_idx,
                neighbor_points_idx,
                systole_nodes,
                systole_normals[node_idx],
            )

            # calculate the least squares solution
            # NOTE: starting with no weighting...
            deformation_gradient = weighted_least_squares(
                diastolic_matrix,
                systolic_vector,
                np.ones(diastolic_matrix.shape[0]),
            )

            deformation_gradient = deformation_gradient.reshape(3, 3, order="F")
            def_gradient_determinant[node_idx] = np.linalg.det(deformation_gradient)

            circumferential_normal = find_circumferential_direction(
                centerline_points,
                centerline_normals,
                diastole_normals[node_idx],
                diastole_nodes[node_idx],
            )
            circumferential_strain[node_idx] = np.linalg.norm(np.dot(deformation_gradient, circumferential_normal)) ** 2
            # node_stiffness_wall_proper[node_idx] = pressure_delta[node_idx] / (circumferential_strain[node_idx] - 1)
            # node_stiffness_wall[node_idx] = pressure_delta[node_idx] / circumferential_strain[node_idx]
            # systolic_node_normals[node_idx] = systolic_node_normal
            systolic_circumferential_normals[node_idx] = circumferential_normal

        def_gradient_determinant[def_gradient_determinant == 0] = np.nan
        circumferential_strain[circumferential_strain == 0] = np.nan
        node_stiffness_wall[node_stiffness_wall == 0] = np.nan
        systolic_node_normals[systolic_node_normals[:, 0] == 0, :] = np.nan
        systolic_circumferential_normals[systolic_circumferential_normals[:, 0] == 0, :] = np.nan

        results = pv.PolyData(diastole_nodes)

        # maybe just add to existing dataset but then save as something new...
        vdm.point_data.set_vectors(systolic_circumferential_normals, "Systolic Circumferential Direction")
        vdm.point_data.set_scalars(def_gradient_determinant, "|Deformation Gradient|")
        vdm.point_data.set_scalars(circumferential_strain - 1, "Circumferential Strain")
        # vdm.point_data.set_scalars(node_stiffness_wall_proper, "Effective Stiffness")
        # vdm.point_data.set_scalars(node_stiffness_wall, "Node Stiffness Wall")
        vdm.point_data.set_vectors(systolic_node_normals, "Systolic Radial Direction")
        vdm.point_data.set_vectors(diastole_normals, "Diastolic Radial Direction")

        vdm.save(f"VDM_modified_{neighbor_extent}.vtp")


if __name__ == "__main__":
    main()
