import numpy as np
import trimesh
import transforms3d

# import pyrender
from pyrender_dev import pyrender
import matplotlib.pyplot as plt
from build_depth_surface import *

# fuze_trimesh = trimesh.load('../models/CalibrationCube_10mm_5mm.stl')
# trimesh1 = trimesh.load('../models/fuzzybear100kpoly.stl')

# trimesh1 = trimesh.load('../models/nut_6x9.stl')
trimesh1 = trimesh.load('../models/car_test.stl')

# trimesh1 = trimesh.load('../models/CalibrationCube_10mm.stl')
# trimesh2 = trimesh.load('../models/CalibrationCube_20mm.stl')

bb = trimesh1.bounding_box

print(bb.extents)
print(trimesh1.bounds)
print(bb.bounds)

mesh1 = pyrender.Mesh.from_trimesh(trimesh1)
mesh2 = pyrender.Mesh.from_trimesh(bb)

bounding_box = [200.0, 200.0, 200.0]

# depth = render_depth_map(mesh1, bounding_box, direction='top', znear=0.05)
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

# depth_surface = build_depth_surface_mesh(depth, bounding_box)
# print(depth_surface)

scene = pyrender.Scene()
scene.add(mesh1)
scene.add(mesh2)

pyrender.Viewer(scene, use_raymond_lighting=True)

# plt.figure()
# plt.subplot(1,2,1)
# plt.axis('off')
# plt.imshow(color)
# plt.subplot(1,2,2)
# plt.axis('off')
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

# print(depth)