import sys
import trimesh
# from pyrender_dev import pyrender
import pyrender
# fuze_trimesh = trimesh.load('../models/CalibrationCube_10mm_5mm.stl')

# tmesh = trimesh.load('../models/car_test.stl')
# fuze_trimesh = trimesh.load('../models/Fennec_Fox.stl')

# tmesh = trimesh.load('../ModelNet10/toilet/train/toilet_0003.off')

# tmesh = trimesh.load('../ModelNet10/dresser/train/dresser_0200.off')

if len(sys.argv) >= 2:
  filename = sys.argv[1]
else:
  filename = '../models/02876657/1b64b36bf7ddae3d7ad11050da24bb12/model.obj'

tmesh = trimesh.load(filename)

print(tmesh.bounds)

mesh = pyrender.Mesh.from_trimesh(tmesh)
scene = pyrender.Scene()
scene.add(mesh)
# pyrender.Viewer(scene, use_raymond_lighting=True)

pyrender.Viewer(scene, use_raymond_lighting=True)