from shrink_wrap_quad_mesh import *
from build_depth_surface import *
from pyrender_dev import pyrender
import matplotlib.pyplot as plt
import trimesh
import transforms3d

# Params
target = trimesh.creation.uv_sphere(radius=1.0)
print(target.bounds)
# target = trimesh.load('../models/cars/car_thingi_whole.stl')
# car_thingi_whole
# target = trimesh.load('../models/fuzzybear100kpoly.stl')
# target = trimesh.load('../ModelNet10/dresser/train/dresser_0020.off')
# target = trimesh.load('../ModelNet10/dresser/train/dresser_0020.off')
previewModel = False
previewDepthMap = False
previewDepthSurfaces = False
previewEachSubdivPass = True

def preview_mesh(m):
  scene = pyrender.Scene()
  scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

def preview_meshes(meshes):
  scene = pyrender.Scene()
  for m in meshes:
    scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

def preview_ray_tracing(meshes, points):
  scene = pyrender.Scene()
  for m in meshes:
    scene.add(pyrender.Mesh.from_trimesh(m))

  sm = trimesh.creation.uv_sphere(radius=0.02)
  sm.visual.vertex_colors = [1.0, 0.0, 0.0]
  tfs = np.tile(np.eye(4), (len(points), 1, 1))
  tfs[:,:3,3] = points

  pm = pyrender.Mesh.from_trimesh(sm, poses=tfs)
  scene.add(pm)

  pyrender.Viewer(scene, use_raymond_lighting=True)


# Init
target_mesh = pyrender.Mesh.from_trimesh(target)
# print(target.bounds)
# bounding_box = [50.0, 50.0, 50.0]
# bounding_box = [400.0, 200.0, 100.0]

bounding_box = [3.0, 3.0, 3.0]

resolution_multiplier = 20


# depth = render_depth_map(mesh1, bounding_box, direction='top', znear=0.05)
# plt.imshow(depth, cmap=plt.cm.gray_r)
# plt.show()

# depth_surface = build_depth_surface_mesh(depth, bounding_box)

# mesh_transform = transforms3d.affines.compose([0, 0, -target.bounds[0,2]], np.eye(3), np.ones(3))
mesh_transform = transforms3d.affines.compose([0, 0, 0], np.eye(3), np.ones(3))
# mesh_bounds = target.bounds + [0, 0, -target.bounds[0,2]]

# mesh_bounds = np.array([[-0.7, -0.7, -0.7], [0.7, 0.7, 0.7]])
mesh_bounds = np.array([[-0.7, -0.7, -0.7], [0.7, 0.7, 0.7]])

# initial_box = 

if previewModel:
  preview_mesh(target)

map_top, s_top = build_depth_surface(target_mesh, bounding_box, direction='top', znear=0.05, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
map_front, s_front = build_depth_surface(target_mesh, bounding_box, direction='front', znear=0.05, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
map_back, s_back = build_depth_surface(target_mesh, bounding_box, direction='back', znear=0.05, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
map_left, s_left = build_depth_surface(target_mesh, bounding_box, direction='left', znear=0.05, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
map_right, s_right = build_depth_surface(target_mesh, bounding_box, direction='right', znear=0.05, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
# build s_bottom
s_bottom = build_plane([bounding_box[0], bounding_box[1]], axes='xy')

depth_surfaces = [s_top, s_front, s_back, s_left, s_right, s_bottom]

if previewDepthMap:
  for dmap in [map_top, map_front, map_back, map_left, map_right]:
    plt.imshow(dmap, cmap=plt.cm.gray_r)
    plt.show()

if previewDepthSurfaces:
  for s in [s_top, s_front, s_back, s_left, s_right, s_bottom]:
    preview_mesh(s)

  # depth_preview_scene = pyrender.Scene()
  # # depth_preview_scene.add(target_mesh)
  # depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_top))
  # depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_front))
  # depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_back))
  # depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_left))
  # depth_preview_scene.add(pyrender.Mesh.from_trimesh(s_right))
  # pyrender.Viewer(depth_preview_scene, use_raymond_lighting=True)

# create initialize quad
bb = ShrinkWrapQuadMesh.from_box(bounds=mesh_bounds)
quad_mesh = ShrinkWrapQuadMesh.from_box(bounds=mesh_bounds)

# preview_meshes([target, bb.get_tri_mesh()])

indices = range(len(quad_mesh.vertices))
# indices = [0, 1]

# Initial Mesh
distances, points = quad_mesh.get_min_projection_distances_to_surfaces(indices, [target, s_bottom])
preview_ray_tracing([target, s_bottom], points)

quad_mesh.project_vertices(indices, distances)

for i in range(5):
  new_vertices = quad_mesh.subdivide()
  d, p = quad_mesh.get_min_projection_distances_to_surfaces(new_vertices, depth_surfaces)
  quad_mesh.project_vertices(new_vertices, d)

  print('subdiv', i)
  if previewEachSubdivPass:
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


# tri_mesh = quad_mesh.get_tri_mesh()

# Preview
# mesh = pyrender.Mesh.from_trimesh(tri_mesh)
