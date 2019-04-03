import trimesh
# from pyrender_dev import pyrender
import pyrender
import numpy as np
# fuze_trimesh = trimesh.load('../models/CalibrationCube_10mm_5mm.stl')
tm = trimesh.load('../models/car_test.stl')
# fuze_trimesh = trimesh.load('../models/Fennec_Fox.stl')
mesh = pyrender.Mesh.from_trimesh(tm)
scene = pyrender.Scene()
# pyrender.Viewer(scene, use_raymond_lighting=True)

pts = tm.vertices.copy()
sm = trimesh.creation.uv_sphere(radius=0.1)
sm.visual.vertex_colors = [1.0, 0.0, 0.0]
tfs = np.tile(np.eye(4), (len(pts), 1, 1))
tfs[:,:3,3] = pts
# m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

colors = np.random.uniform(size=pts.shape)
m = pyrender.Mesh.from_points(pts, colors=colors)

scene.add(m)


pyrender.Viewer(scene, use_raymond_lighting=False)