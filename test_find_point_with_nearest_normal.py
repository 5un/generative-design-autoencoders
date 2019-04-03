import numpy as np
import trimesh
from pyrender_dev import pyrender
from shrink_wrap_quad_mesh import *

def preview_mesh(m):
  scene = pyrender.Scene()
  scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

bounds = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
quad_box = ShrinkWrapQuadMesh.from_box(bounds=bounds)
box = trimesh.creation.box([1,1,1])
sphere = trimesh.creation.uv_sphere(10.0)

print(box.vertex_normals)

preview_mesh(quad_box.get_tri_mesh())
# preview_mesh(sphere)