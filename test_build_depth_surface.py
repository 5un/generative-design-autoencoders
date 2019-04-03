import numpy as np
import trimesh
import transforms3d
# import pyrender
from pyrender_dev import pyrender
import matplotlib.pyplot as plt
from build_depth_surface import *

showDepthImagePreview = True
showIsolatedDepthSurfacePreview = True

def preview_mesh(m, original_mesh=None):
  scene = pyrender.Scene()
  scene.add(pyrender.Mesh.from_trimesh(m))
  if original_mesh is not None:
    scene.add(pyrender.Mesh.from_trimesh(original_mesh))
  pyrender.Viewer(scene, use_raymond_lighting=True)

# fuze_trimesh = trimesh.load('../models/CalibrationCube_10mm_5mm.stl')
# trimesh1 = trimesh.load('../models/fuzzybear100kpoly.stl')

# trimesh1 = trimesh.load('../models/nut_6x9.stl')
trimesh1 = trimesh.load('../models/Fennec_Fox.stl')

# trimesh1 = trimesh.load('../models/CalibrationCube_10mm.stl')
trimesh2 = trimesh.load('../models/CalibrationCube_20mm.stl')

mesh1 = pyrender.Mesh.from_trimesh(trimesh1)
mesh2 = pyrender.Mesh.from_trimesh(trimesh2)

# bounding_box = [[-75.0, -100.0, -50.0], [75.0, 100.0, 50.0]]

bounding_box = [[-30.0, -60.0, 0], [30.0, 60.0, 100]]

# bounding_box = [[-55.0, -80.0, 0], [75.0, 100.0, 100]]

# depth = render_depth_map(mesh1, bounding_box, direction='top', znear=0.05)
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

# depth_surface = build_depth_surface_mesh(depth, bounding_box)
# print(depth_surface)

rm = 4

d_top, s_top = build_depth_surface(mesh1, bounding_box, direction='+z', znear=0.01, resolution_multiplier=rm)
d_bottom, s_bottom = build_depth_surface(mesh1, bounding_box, direction='-z', znear=0.01, resolution_multiplier=rm)
d_front, s_front = build_depth_surface(mesh1, bounding_box, direction='-y', znear=0.01, resolution_multiplier=rm)
d_back, s_back = build_depth_surface(mesh1, bounding_box, direction='+y', znear=0.01, resolution_multiplier=rm)
d_left, s_left = build_depth_surface(mesh1, bounding_box, direction='+x', znear=0.01, resolution_multiplier=rm)
d_right, s_right = build_depth_surface(mesh1, bounding_box, direction='-x', znear=0.01, resolution_multiplier=rm)

if showDepthImagePreview:
  for dmap in [d_top, d_bottom, d_front, d_back, d_left, d_right]:
    plt.imshow(dmap, cmap=plt.cm.gray_r)
    plt.show()

if showIsolatedDepthSurfacePreview:
  for s in [s_top, s_bottom, s_front, s_back, s_left, s_right]:
    preview_mesh(s)

depth_preview_scene = pyrender.Scene()
depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_top))
depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_bottom))
depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_front))
depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_back))
depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_left))
depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_right))

pyrender.Viewer(depth_preview_scene, use_raymond_lighting=True)
