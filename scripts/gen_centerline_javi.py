#!/usr/bin/env python
"""Created on 2024/10/02 15:18:53

@author: Javiera Jilberto Vallejos
"""

import cheartio as chio
import meshio as io
import numpy as np
import pyvista as pv
from scipy.interpolate import interp1d


def get_normal_plane_svd(points):  # Find the plane that minimizes the distance given N points
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


mesh = chio.read_mesh("model", meshio=True)
bdata = chio.read_bfile("model", element="tetra")
lap = chio.read_dfile("lap.FE") * (1 + 2e-3) - 1e-3

ien = mesh.cells[0].data

# Get endo surface
tri_endo = bdata[bdata[:, -1] == 3, 1:-1]

tri_mesh = io.Mesh(points=mesh.points, cells=[("triangle", tri_endo)], point_data={"lap": lap})
io.write("endo.vtu", tri_mesh)

meshio_mesh = mesh
mesh = pv.from_meshio(tri_mesh)

ndiv = 21
div_values = np.linspace(0, 1, ndiv)

contours = []
centers = []
values = []
normals = []
for val in div_values:
    contour = mesh.contour(isosurfaces=[val])

    points = contour.points

    contours.append(points)
    if len(points) == 0:
        continue

    values.append(val)

    center = np.mean(points, axis=0)
    centers.append(center)

    # Find normal to the contour
    normal = get_normal_plane_svd(points)[0]
    normals.append(normal)

# Flip normals if needed
for i in range(1, len(normals)):
    if np.dot(normals[i], normals[i - 1]) < 0:
        normals[i] = -normals[i]

centers = np.vstack(centers)
normals = np.vstack(normals)

center_func = interp1d(values, centers.T, kind="cubic", fill_value="extrapolate")

node_center = np.array(center_func(lap)).T

sampled_lap = np.linspace(0, 1, 100)
sampled_centers = center_func(sampled_lap).T
sampled_normals = np.diff(sampled_centers, axis=0)
sampled_normals = np.vstack([sampled_normals, sampled_normals[-1]])

normal_func = interp1d(sampled_lap, sampled_normals.T, kind="cubic", fill_value="extrapolate")

node_normals = np.array(normal_func(lap)).T
node_normals = node_normals / np.linalg.norm(node_normals, axis=1)[:, None]

# Find centroid and aorta plane
centroid = np.mean(np.vstack(node_center), axis=0)

# Grab surface nodes to calculate the aorta plane (using all the nodes it's too much memory)
surface_points = mesh.points[np.unique(bdata[bdata[:, -1] == 3, 1:-1])]

aorta_plane, _ = get_normal_plane_svd(surface_points)
aorta_plane = aorta_plane / np.linalg.norm(aorta_plane)

# Find the vector that goes from the center to the centroid
aux = node_center - centroid
aux = aux / np.linalg.norm(aux, axis=0)

# Calculate out of plane vector
ex = np.cross(aux, node_normals, axisa=1, axisb=1)
ex = ex / np.linalg.norm(ex, axis=0)

node_ex = np.tile(aorta_plane, (node_normals.shape[0], 1))


# Calculate ex
ey = np.cross(node_ex, node_normals, axisa=1, axisb=1)
node_ey = ey / np.linalg.norm(ey, axis=1)[:, None]

node_vector = mesh.points - node_center

node_vx = np.sum(node_vector * node_ex, axis=1)
node_vy = np.sum(node_vector * node_ey, axis=1)
node_theta = np.arctan2(node_vx, node_vy)
node_theta2 = np.arctan2(node_vy, node_vx)
print(node_theta.shape)


# centroid = np.mean(np.vstack(centers), axis=0)

# exs = []
# eys = []
# for i in range(len(centers)):
#     # Find the vector that goes from the center to the centroid
#     aux = centers[i] - centroid
#     aux = aux/np.linalg.norm(aux)

#     # Calculate out of plane vector
#     ex = np.cross(aux, normals[i])
#     ex = ex/np.linalg.norm(ex)

#     # Calculate ex
#     ey = np.cross(ex, normals[i])
#     ey = ey/np.linalg.norm(ey)

#     exs.append(ex)
#     eys.append(ey)

# values = np.array(values)
# centers = np.vstack(centers)
# normals = np.vstack(normals)
# exs = np.vstack(exs)
# eys = np.vstack(eys)

# print('Creating interpolation functions')
# center_func = interp1d(values, centers.T, kind='cubic', fill_value='extrapolate')
# normal_func = interp1d(values, normals.T, kind='cubic', fill_value='extrapolate')
# print('Interpolating ex')
# ex_func = interp1d(values, exs.T, kind='cubic', fill_value='extrapolate')
# print('Interpolating ey')
# ey_func = interp1d(values, eys.T, kind='cubic', fill_value='extrapolate')

# print('Interpolating center')
# node_center = np.array(center_func(lap)).T
# print('Interpolating ex')
# node_ex = np.array(ex_func(lap)).T
# print('Interpolating ey')
# node_ey = np.array(ey_func(lap)).T
# node_vector = mesh.points - node_center[:,None]

# print('Calculating theta')
# node_vx = np.sum(node_vector*node_ex, axis=0)
# node_vy = np.sum(node_vector*node_ey, axis=0)
# node_theta = np.arctan2(node_vy, node_vx)

# print('done!')
# chio.write_dfile('center.FE', node_center)

print("saving mesh")
# meshio_mesh.point_data['center'] = node_center
# meshio_mesh.point_data['normal'] = node_normals
# meshio_mesh.point_data['ex'] = node_ex
# meshio_mesh.point_data['ey'] = node_ey
# meshio_mesh.point_data['aux'] = aux
node_theta = node_theta / np.pi
node_theta2 = node_theta2 / np.pi
meshio_mesh.point_data["circ"] = node_theta
meshio_mesh.point_data["circ2"] = node_theta2
meshio_mesh.point_data["circ_cos"] = np.cos(node_theta)
meshio_mesh.point_data["circ_sin"] = np.sin(node_theta)

chio.write_dfile("circumferential.FE", node_theta)

basis = io.read("basis.vtu")
meshio_mesh.point_data["long"] = basis.point_data["long"]
meshio_mesh.point_data["trans"] = basis.point_data["trans"]

io.write("center.vtu", meshio_mesh)

# print('saving centerline')
centers = center_func(np.linspace(0, 1, 100)).T
normals = normal_func(np.linspace(0, 1, 100)).T
lines = np.array([np.arange(len(centers)), np.arange(1, len(centers) + 1)]).T
io.write_points_cells("centerline.vtu", centers, {"line": lines}, point_data={"normal": normals})
# io.write_points_cells('contours.vtu', np.vstack(contours), {'vertex': np.arange(len(np.vstack(contours)))[:,None]})


# Make cuts for plots
import pymmg

abs_circ2 = np.abs(node_theta2)
new_mesh = pymmg.mmg_isoline_meshing(meshio_mesh, abs_circ2, 0.4, funcs_to_interpolate=["circ", "long", "trans"])
f = new_mesh.point_data["f"]
f_elem = np.mean(f[new_mesh.cells[0].data], axis=1)
new_mesh.cell_data["f"] = [(f_elem > 0.4).astype(int)]
io.write("center_cuts.vtu", new_mesh)
