from shrink_wrap_quad_mesh import *
from pyrender_dev import pyrender
import trimesh

target = trimesh.load('../models/car_test.stl')

# create initialize quad
quad_mesh = ShrinkWrapQuadMesh.from_box(bounds=target.bounds)

def preview_mesh(m):
  scene = pyrender.Scene()
  scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

preview_mesh(quad_mesh.get_tri_mesh())

for i in range(5):
  # distances, points = quad_mesh.get_projection_distances(range(len(quad_mesh.vertices)), target, inverted=True)
  # print(distances)
  new_vertices = quad_mesh.subdivide()
  quad_mesh.project_vertices(new_vertices, np.repeat(10.0, len(quad_mesh.vertices)))
  preview_mesh(quad_mesh.get_tri_mesh())

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