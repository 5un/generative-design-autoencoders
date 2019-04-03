import numpy as np
import trimesh
from pyrender_dev import pyrender
import transforms3d

def render_depth_map(mesh, bounding_box, direction='top', znear=0.05, resolution_multiplier=1, mesh_transform=None):

  scene = pyrender.Scene()
  
  x_mid = (bounding_box[1][0] + bounding_box[0][0]) / 2.0
  y_mid = (bounding_box[1][1] + bounding_box[0][1]) / 2.0
  z_mid = (bounding_box[1][2] + bounding_box[0][2]) / 2.0

  if mesh_transform is not None:
    scene.add(mesh, pose=mesh_transform)
  else:
    scene.add(mesh)
  
  # +Z
  if direction == 'top':
    zfar = bounding_box[2]
    im_width = bounding_box[0]
    im_height = bounding_box[1]
    camera_pose = transforms3d.affines.compose([0, 0, zfar], np.eye(3), np.ones(3))

  # if direction == 'bottom':
  #   zfar = bounding_box[2]
  #   im_width = bounding_box[0]
  #   im_height = bounding_box[1]
  #   camera_pose = transforms3d.affines.compose([0, 0, zfar], np.eye(3), np.ones(3))

  # -Y
  elif direction == 'front':
    zfar = bounding_box[1]
    im_width = bounding_box[0]
    im_height = bounding_box[2]

    m1 = transforms3d.axangles.axangle2mat((1.0, 0.0, 0.0), np.radians(90))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.affines.compose([0, -zfar * 0.5, bounding_box[2]], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m2, m1)

  # +Y (must also rotate around z again)
  elif direction == 'back':
    zfar = bounding_box[1]
    im_width = bounding_box[0]
    im_height = bounding_box[2]

    m1 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(180))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.axangles.axangle2mat((1.0, 0.0, 0.0), np.radians(-90))
    m2 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))
    m3 = transforms3d.affines.compose([0, zfar * 0.5, bounding_box[2]], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m3, np.matmul(m2, m1))

  # +X
  elif direction == 'left':
    zfar = bounding_box[0]
    im_width = bounding_box[1]
    im_height = bounding_box[2]

    # rotate cam around y?
    m1 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(90))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.axangles.axangle2mat((0.0, 1.0, 0.0), np.radians(90))  
    m2 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))
    m3 = transforms3d.affines.compose([zfar * 0.5, 0, bounding_box[2]], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m3, np.matmul(m2, m1))

  # -X
  elif direction == 'right':
    zfar = bounding_box[0]
    im_width = bounding_box[1]
    im_height = bounding_box[2]

    # rotate cam around y?
    m1 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(-90))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.axangles.axangle2mat((0.0, 1.0, 0.0), np.radians(-90))
    m2 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))
    m3 = transforms3d.affines.compose([-zfar * 0.5, 0, bounding_box[2]], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m3, np.matmul(m2, m1))

  viewport_width = im_width
  viewport_height = im_height
  camera = pyrender.OrthographicCamera(xmag=viewport_width, ymag=viewport_height, znear=znear, zfar=zfar)

  # -Z
  # x_rot = np.radians(180)
  # camera_pose = np.array([
  #   [1.0, 0.0, 0.0, 0.0],
  #   [0.0, np.cos(x_rot), -np.sin(x_rot), 0.0],
  #   [0.0, np.sin(x_rot), np.cos(x_rot), -100.0],
  #   [0.0, 0.0, 0.0, 1.0],
  # ])

  scene.add(camera, pose=camera_pose)
  # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
  # scene.add(light, pose=camera_pose)

  r = pyrender.OffscreenRenderer(im_width * resolution_multiplier, im_height * resolution_multiplier)
  color, depth = r.render(scene)
  # plt.imshow(color)
  # plt.imshow(depth)

  # Remap the depth
  depth = depth * (zfar - znear) + znear

  return depth


def build_depth_surface_mesh(depth_img, bounding_box, direction):
  """
  Parameters
  ----------
  depth_image: ndarray
  """
  vertices = []
  faces = []

  w = depth_img.shape[1]
  h = depth_img.shape[0]

  print(depth_img.shape)

  if direction == 'top':
    c_cell_size = bounding_box[0] / w * 2
    r_cell_size = bounding_box[1] / h * 2
  elif direction == 'front' or direction == 'back':
    c_cell_size = bounding_box[0] / w * 2
    r_cell_size = bounding_box[2] / h * 2
  elif direction == 'left' or direction == 'right':
    c_cell_size = bounding_box[1] / w * 2
    r_cell_size = bounding_box[2]  / h * 2

  for r in range(h):
    for c in range(w):
      if direction == 'top':
        vertices.append(-(bounding_box[0]) +  c * c_cell_size)
        vertices.append((bounding_box[1]) - (r * r_cell_size))
        vertices.append(bounding_box[2] - depth_img[r, c])
      elif direction == 'front':
        vertices.append(-(bounding_box[0]) +  c * c_cell_size)
        vertices.append(-bounding_box[1]*0.5 + depth_img[r, c])
        vertices.append((bounding_box[2] * 2) - (r * r_cell_size))
      elif direction == 'back':
        vertices.append((bounding_box[0]) -  c * c_cell_size)
        vertices.append(bounding_box[1]*0.5 - depth_img[r, c])
        vertices.append((bounding_box[2] * 2) - (r * r_cell_size))
      elif direction == 'left':
        # vertices.append(-bounding_box[0]*0.5 + depth_img[r, c])
        vertices.append(bounding_box[0]*0.5 - depth_img[r, c])
        vertices.append(-(bounding_box[1]) +  c * c_cell_size)
        vertices.append((bounding_box[2] * 2) - (r * r_cell_size))
      elif direction == 'right':
        #vertices.append(bounding_box[0]*0.5 - depth_img[r, c])
        vertices.append(-bounding_box[0]*0.5 + depth_img[r, c])
        vertices.append((bounding_box[1]) - c * c_cell_size)
        vertices.append((bounding_box[2] * 2) - (r * r_cell_size))

        

      if (c + 1 < w) and (r + 1 < h):
        idx = (r * w) + c
        # faces.append(idx)
        # faces.append(idx + 1)
        # faces.append(idx + w + 1)
        # faces.append(idx)
        # faces.append(idx + w + 1)
        # faces.append(idx + w)

        faces.append(idx)
        faces.append(idx + w + 1)
        faces.append(idx + 1)
        faces.append(idx)
        faces.append(idx + w)
        faces.append(idx + w + 1)

  vertices = np.array(vertices,
                      order='C',
                      dtype=np.float64).reshape((-1, 3))

  faces = np.array(faces,
                   order='C', dtype=np.int64).reshape((-1, 3))

  depth_surface = trimesh.Trimesh(vertices=vertices,
                faces=faces,
                # face_normals=face_normals,
                process=False)

  return depth_surface

def build_depth_surface(mesh, bounding_box, direction='top', znear=0.05, resolution_multiplier=1, mesh_transform=None):
  depth = render_depth_map(mesh, bounding_box, direction, znear, resolution_multiplier, mesh_transform)
  s = build_depth_surface_mesh(depth, bounding_box, direction)
  return depth, s

def build_depth_surfaces(mesh, bounding_box, znear=0.05):
  # build surfaces in all directions
  return []

def build_plane(dimensions, axes='xy'):
  vertices = []
  faces = []

  if axes == 'xy':
    vertices = [-dimensions[0], -dimensions[1], 0,
                  dimensions[0], -dimensions[1], 0,
                  dimensions[0], dimensions[1], 0,
                  -dimensions[0], dimensions[1], 0]
  elif axes == 'xz':
    vertices = [-dimensions[0], 0, -dimensions[1],
                  dimensions[0], 0, -dimensions[1],
                  dimensions[0], 0, dimensions[1],
                  -dimensions[0], 0, dimensions[1]]
  elif axes == 'yz':
    vertices = [0, -dimensions[0], -dimensions[1],
                0, dimensions[0], -dimensions[1],
                0, dimensions[0], dimensions[1],
                0, -dimensions[0], dimensions[1]]

  faces = [0, 1, 2, 0, 2, 3]

  vertices = np.array(vertices,
                      order='C',
                      dtype=np.float64).reshape((-1, 3))

  faces = np.array(faces,
                   order='C', dtype=np.int64).reshape((-1, 3))

  plane = trimesh.Trimesh(vertices=vertices,
                faces=faces,
                process=False)
  return plane



