import cheartio
import h5py
import meshio
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


def load_data(path: str, systole_timestep: int):
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
    # holy bonobos
    smoothed_centerline = pv.read("new_points.vtp")
    centerline_points = smoothed_centerline.points
    centerline_normals = smoothed_centerline.point_data["Derivative"]

    vdm = pv.read("UM16_VDM-D.vtp")
    diastole_nodes = vdm.points
    diastole_normals = vdm.point_data["SurfaceNormals"]  # I really don't know if this is the diastolic normal or not
    displacement = vdm.point_data["Displacement"]
    systole_nodes = diastole_nodes + displacement

    ###############################################################################
    # read in geometry and data
    xyz, bfile, systole_d, diastole_d = load_data("model", 241)
    node_normals = cheartio.read_dfile("norms.D")

    # calculate displacement field
    displacement = systole_d["X"] - diastole_d["X"]

    with h5py.File("centerline.h5", "r") as f:
        centerline_points = np.asarray(f["centerline_points"])
        centerline_normals = np.asarray(f["centerline_normals"][:])

    # find all nodes that exist on outer wall; flatten and take unique values
    wall_bfile_elements = bfile[np.isin(bfile[:, 4], 8), 1:-1]  # <- this has original indexing from X file...
    wall_bfile_nodes = np.unique(wall_bfile_elements.flatten())

    # find nodes that have moved at all (1e-7 tolerance)
    moving_nodes = np.abs(np.linalg.norm(displacement, axis=1)) > 1e-7
    # instead of having a giant true/false array, we are going to grab the row indices of the moving nodes
    moving_nodes_idx = np.arange(displacement.shape[0])[moving_nodes]

    moving_wall_bfile_elements = np.isin(wall_bfile_elements, moving_nodes_idx)
    moving_wall_elements = wall_bfile_elements[np.all(moving_wall_bfile_elements, axis=1)]

    # find nodes on the outer wall boundary that have also moved (pretty much all except for boundary edges)
    moving_wall_nodes = np.intersect1d(wall_bfile_nodes, moving_nodes_idx)
    # this turns out to be the same as finding all of the unique elements in moving_wall_elements

    # NOTE we do not want the displacements per se, but we will still use this particular indexing pattern for the diastolic and systolic data
    systole_wall = systole_d["X"][moving_wall_nodes]
    diastole_wall = diastole_d["X"][moving_wall_nodes]

    node_normals_restricted = node_normals[moving_wall_nodes]

    pressure_delta = systole_d["P"] - diastole_d["P"]
    pressure_delta_outer_wall = pressure_delta[moving_wall_nodes]

    ################## Calculate Area Stiffness ##################
    # 1a) Calculate strain for each triangle on the outer wall
    # note that every triangle WILL receive a strain value, unlike the displacement method where edges nodes are NaN

    cells = [("triangle", moving_wall_elements)]
    # 1 means adjacent triangles, 2 means neighbors of neighbors, 3 means neighbors of neighbors of neighbors...
    for neighbor_extent in range(1, 5):
        def_gradient_determinant = np.zeros(xyz.shape[0])
        circumferential_strain = np.zeros(xyz.shape[0])
        node_stiffness_wall = np.zeros(xyz.shape[0])
        node_stiffness_wall_proper = np.zeros(xyz.shape[0])
        systolic_node_normals = np.zeros((xyz.shape[0], 3))
        systolic_circumferential_normals = np.zeros((xyz.shape[0], 3))
        for node_idx in np.unique(moving_wall_elements.flatten()):
            # find idx of all connected elements

            # FIND INITIAL CONNECTED TRIANGLES
            triangle_idx = np.argwhere(np.any(moving_wall_elements == node_idx, axis=1)).flatten()

            for _ in range(neighbor_extent - 1):
                connected_nodes = np.unique(moving_wall_elements[triangle_idx].flatten())
                connected_triangles = []
                for node in connected_nodes:
                    connected_triangles.extend(
                        np.argwhere(np.any(moving_wall_elements == node, axis=1)).flatten().tolist(),
                    )

                triangle_idx = np.unique(np.array(connected_triangles).flatten())

            # NOW we have all triangles that we'd like to look at...
            unique_neighbor_node_idx = np.setdiff1d(np.unique(moving_wall_elements[triangle_idx]).flatten(), node_idx)

            # SO WE ACTUALLY NEED TO GET OUR NORMALS HERE...
            systolic_node_normal = estimate_normal(
                node_idx,
                node_normals,
                moving_wall_elements[triangle_idx],
                systole_d["X"],
            )

            diastolic_matrix = construct_diastolic_matrix(
                node_idx,
                unique_neighbor_node_idx,
                xyz,
                node_normals[node_idx] / np.linalg.norm(node_normals[node_idx]),
            )
            systolic_vector = construct_systolic_vector(
                node_idx,
                unique_neighbor_node_idx,
                systole_d["X"],
                systolic_node_normal,
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
                node_normals[node_idx],
                xyz[node_idx],
            )
            circumferential_strain[node_idx] = np.linalg.norm(np.dot(deformation_gradient, circumferential_normal)) ** 2
            node_stiffness_wall_proper[node_idx] = pressure_delta[node_idx] / (circumferential_strain[node_idx] - 1)
            node_stiffness_wall[node_idx] = pressure_delta[node_idx] / circumferential_strain[node_idx]
            systolic_node_normals[node_idx] = systolic_node_normal
            systolic_circumferential_normals[node_idx] = circumferential_normal

        # node_strain_connected = (node_area_sys_connected - node_area_dia_connected) / node_area_dia_connected
        # node_stiffness_connected = pressure_delta / node_strain_connected
        # node_stiffness_wall = pressure_delta[moving_wall_nodes] / node_strain_connected[moving_wall_nodes]

        def_gradient_determinant[def_gradient_determinant == 0] = np.nan
        circumferential_strain[circumferential_strain == 0] = np.nan
        node_stiffness_wall[node_stiffness_wall == 0] = np.nan
        systolic_node_normals[systolic_node_normals[:, 0] == 0, :] = np.nan
        systolic_circumferential_normals[systolic_circumferential_normals[:, 0] == 0, :] = np.nan

        meshio.write_points_cells(
            f"visualizations/estimated_stiffness_circumferential_k{neighbor_extent}.vtu",
            xyz,
            cells,
            point_data={
                "|Deformation Gradient|": def_gradient_determinant,
                "point_normals": node_normals,
                "circumferential_strain": circumferential_strain,
                "circumferential_strain_proper": circumferential_strain - 1,
                "node_stiffness_wall": node_stiffness_wall,
                "node_stiffness_wall_proper": node_stiffness_wall_proper,
                "systolic_node_normals": systolic_node_normals,
                "systolic_circumferential_normals": systolic_circumferential_normals,
            },
        )


if __name__ == "__main__":
    main()
