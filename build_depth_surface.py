import numpy as np
import trimesh
from pyrender_dev import pyrender
import transforms3d

def render_depth_map(mesh, bounding_box, direction='top', znear=0.05, resolution_multiplier=1, mesh_transform=None):

  x_min = bounding_box[0][0]
  y_min = bounding_box[0][1]
  z_min = bounding_box[0][2]
  x_max = bounding_box[1][0]
  y_max = bounding_box[1][1]
  z_max = bounding_box[1][2]
  x_mid = (bounding_box[1][0] + bounding_box[0][0]) / 2.0
  y_mid = (bounding_box[1][1] + bounding_box[0][1]) / 2.0
  z_mid = (bounding_box[1][2] + bounding_box[0][2]) / 2.0
  x_range = bounding_box[1][0] - bounding_box[0][0]
  y_range = bounding_box[1][1] - bounding_box[0][1]
  z_range = bounding_box[1][2] - bounding_box[0][2]

  scene = pyrender.Scene()

  if mesh_transform is not None:
    scene.add(mesh, pose=mesh_transform)
  else:
    scene.add(mesh)
  
  # +Z
  if (direction == 'top') or (direction == '+z'):
    camera_zfar = z_range
    im_width = x_range
    im_height = y_range

    camera_pose = transforms3d.affines.compose([x_mid, y_mid, z_max], np.eye(3), np.ones(3))

  # -Z
  elif (direction == 'bottom') or (direction == '-z'):
    camera_zfar = z_range
    im_width = x_range
    im_height = y_range
    
    m1 = transforms3d.axangles.axangle2mat((1.0, 0.0, 0.0), np.radians(180))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(180))
    m2 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))
    m3 = transforms3d.affines.compose([x_mid, y_mid, z_min], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m3, np.matmul(m2, m1))

  # -Y
  elif (direction == 'front') or (direction == '-y'):
    camera_zfar = y_range
    im_width = x_range
    im_height = z_range

    m1 = transforms3d.axangles.axangle2mat((1.0, 0.0, 0.0), np.radians(90))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.affines.compose([x_mid, y_min, z_mid], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m2, m1)

  # +Y (must also rotate around z again)
  elif (direction == 'back') or (direction == '+y'):
    camera_zfar = y_range
    im_width = x_range
    im_height = z_range

    m1 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(180))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.axangles.axangle2mat((1.0, 0.0, 0.0), np.radians(-90))
    m2 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))
    m3 = transforms3d.affines.compose([x_mid, y_max, z_mid], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m3, np.matmul(m2, m1))

  # +X
  elif (direction == 'left') or (direction == '+x'):
    camera_zfar = x_range
    im_width = y_range
    im_height = z_range

    # rotate cam around y?
    m1 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(90))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.axangles.axangle2mat((0.0, 1.0, 0.0), np.radians(90))  
    m2 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))
    m3 = transforms3d.affines.compose([x_max, y_mid, z_mid], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m3, np.matmul(m2, m1))

  # -X
  elif (direction == 'right') or (direction == '-x'):
    camera_zfar = x_range
    im_width = y_range
    im_height = z_range

    # rotate cam around y?
    m1 = transforms3d.axangles.axangle2mat((0.0, 0.0, 1.0), np.radians(-90))
    m1 = transforms3d.affines.compose(np.zeros(3), m1, np.ones(3))
    m2 = transforms3d.axangles.axangle2mat((0.0, 1.0, 0.0), np.radians(-90))
    m2 = transforms3d.affines.compose(np.zeros(3), m2, np.ones(3))
    m3 = transforms3d.affines.compose([x_min, y_mid, z_mid], np.eye(3), np.ones(3))
    camera_pose = np.matmul(m3, np.matmul(m2, m1))

  viewport_width = im_width
  viewport_height = im_height
  camera = pyrender.OrthographicCamera(xmag=viewport_width, ymag=viewport_height, znear=znear, zfar=camera_zfar)

  scene.add(camera, pose=camera_pose)
  # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0, innerConeAngle=np.pi/16.0)
  # scene.add(light, pose=camera_pose)

  r = pyrender.OffscreenRenderer(im_width * resolution_multiplier, im_height * resolution_multiplier)
  color, depth = r.render(scene)

  # Remap the depth
  depth = depth * (camera_zfar - znear) + znear

  # TODO: also take care of the laplacian thing

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

  x_min = bounding_box[0][0]
  y_min = bounding_box[0][1]
  z_min = bounding_box[0][2]
  x_max = bounding_box[1][0]
  y_max = bounding_box[1][1]
  z_max = bounding_box[1][2]
  x_mid = (bounding_box[1][0] + bounding_box[0][0]) / 2.0
  y_mid = (bounding_box[1][1] + bounding_box[0][1]) / 2.0
  z_mid = (bounding_box[1][2] + bounding_box[0][2]) / 2.0
  x_range = bounding_box[1][0] - bounding_box[0][0]
  y_range = bounding_box[1][1] - bounding_box[0][1]
  z_range = bounding_box[1][2] - bounding_box[0][2]

  print(depth_img.shape)

  if direction == 'top' or direction == 'bottom' or direction == '+z' or direction == '-z':
    c_cell_size = x_range / w * 2
    r_cell_size = y_range / h * 2 

  elif direction == 'front' or direction == 'back' or direction == '+y' or direction == '-y':
    c_cell_size = x_range / w * 2
    r_cell_size = z_range / h * 2

  elif direction == 'left' or direction == 'right' or direction == '+x' or direction == '-x':
    c_cell_size = y_range / w * 2
    r_cell_size = z_range  / h * 2

  for r in range(h):
    for c in range(w):
      if direction == 'top' or direction == '+z':
        vertices.append(x_min - (x_range * 0.5) +  c * c_cell_size)
        vertices.append(y_max + (y_range * 0.5) - (r * r_cell_size))
        vertices.append(z_max - depth_img[r, c])
      elif direction == 'bottom' or direction == '-z':
        vertices.append(x_max + (x_range * 0.5) -  c * c_cell_size)
        vertices.append(y_max + (y_range * 0.5) - (r * r_cell_size))
        vertices.append(z_min + depth_img[r, c])
      elif direction == 'front' or direction == '-y':
        vertices.append(x_min - (x_range * 0.5) +  c * c_cell_size)
        vertices.append(y_min + depth_img[r, c])
        vertices.append(z_max + (z_range * 0.5) - (r * r_cell_size))
      elif direction == 'back' or direction == '+y':
        vertices.append(x_max + (x_range * 0.5) -  c * c_cell_size)
        vertices.append(y_max - depth_img[r, c])
        vertices.append(z_max + (z_range * 0.5) - (r * r_cell_size))
      elif direction == 'left' or direction == '+x':
        # vertices.append(-bounding_box[0]*0.5 + depth_img[r, c])
        vertices.append(x_max - depth_img[r, c])
        vertices.append(y_min - (y_range * 0.5) +  c * c_cell_size)
        vertices.append(z_max + (z_range * 0.5)  - (r * r_cell_size))
      elif direction == 'right' or direction == '-x':
        #vertices.append(bounding_box[0]*0.5 - depth_img[r, c])
        vertices.append(x_min + depth_img[r, c])
        vertices.append(y_max + (y_range * 0.5) - c * c_cell_size)
        vertices.append(z_max + (z_range * 0.5) - (r * r_cell_size))

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



