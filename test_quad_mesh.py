from shrink_wrap_quad_mesh import *
from pyrender_dev import pyrender
import trimesh

target = trimesh.load('../models/car_test.stl')

# create initialize quad
quad_mesh = ShrinkWrapQuadMesh.from_box(bounds=target.bounds)

distances, points = quad_mesh.get_projection_distances(range(len(quad_mesh.vertices)), target, inverted=True)

quad_mesh.project_vertices(range(len(quad_mesh.vertices)), distances)

# try inverted

# get initialized mesh

# subdiv and do shrink wrap

# keep vertex and project it

# new_vertices = quad_mesh.subdivide()
# print(len(new_vertices))
# print(len(quad_mesh.vertices))

# quad_mesh.project_vertices(new_vertices, np.repeat(0.3, len(new_vertices)))
# quad_mesh.subdivide()
# quad_mesh.subdivide()
# quad_mesh.subdivide()

# print(new_vertices)


tri_mesh = quad_mesh.get_tri_mesh()

# Preview
mesh = pyrender.Mesh.from_trimesh(tri_mesh)
pm = pyrender.Mesh.from_points(points)

scene = pyrender.Scene()
# scene.add(pyrender.Mesh.from_trimesh(target))
scene.add(mesh)
# scene.add(pm)
pyrender.Viewer(scene, use_raymond_lighting=True)