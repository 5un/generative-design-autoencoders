import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt

fuze_trimesh = trimesh.load('./triplescrew_bolt40.stl')
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)

# TODO: figure out how to render corectly

# camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

camera = pyrender.OrthographicCamera(xmag=20.0, ymag=20.0)

# find python lib to calculate it

# s = np.sqrt(2)/2
# camera_pose = np.array([
#   [0.0, -s,   s,   0.3],
#   [1.0,  0.0, 0.0, 0.0],
#   [0.0,  s,   s,   0.35],
#   [0.0,  0.0, 0.0, 1.0],
# ])

# camera_pose = np.array([
#   [1.0, 0.0, 0.0, 0.0],
#   [0.0, 1.0, 0.0, 0.0],
#   [0.0, 0.0, 1.0, 120.0],
#   [0.0, 0.0, 0.0, 1.0],
# ])



x_rot = np.radians(10)

camera_pose = np.array([
  [1.0, 0.0, 0.0, 0.0],
  [0.0, np.cos(x_rot), -np.sin(x_rot), 0.0],
  [0.0, np.sin(x_rot), np.cos(x_rot), 120.0],
  [0.0, 0.0, 0.0, 1.0],
])

scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
# scene.add(light, pose=camera_pose)

r = pyrender.OffscreenRenderer(400, 400)
color, depth = r.render(scene)
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(color)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(depth, cmap=plt.cm.gray_r)
plt.show()

# print(depth)