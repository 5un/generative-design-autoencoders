import numpy as np
import trimesh
from pyrender_dev import pyrender
from build_depth_surface import *
from utils import *
import matplotlib.pyplot as plt

class ShrinkWrapQuadMesh():

  def __init__(self, vertices, faces, face_normals):
    # TODO: look from trimesh
    self.vertices = vertices
    self.faces = faces
    self.face_normals = face_normals

    self.vertex_faces = [[] for x in range(len(vertices))]
    for fi in range(len(faces)):
      for vi in faces[fi]:
        self.vertex_faces[vi].append(fi)

    vertex_normals = []
    for vi in range(len(vertices)):
      v_faces = np.take(self.face_normals, self.vertex_faces[vi], axis=0)
      vn = np.mean(v_faces, axis=0)
      vn = vn / np.sqrt(np.sum(vn **2))
      vertex_normals.append(vn)

    self.vertex_normals = np.array(vertex_normals)

  def from_box(bounds=None, transform=None, **kwargs):
    
    if bounds is None:
      bounds = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]

    # vertices = [
    #   bounds[0, 0], bounds[0, 1], bounds[0, 2],
    #   bounds[0, 0], bounds[0, 1], bounds[1, 2],
    #   bounds[0, 0], bounds[1, 1], bounds[0, 2],
    #   bounds[0, 0], bounds[1, 1], bounds[1, 2],
    #   bounds[1, 0], bounds[0, 1], bounds[0, 2],
    #   bounds[1, 0], bounds[0, 1], bounds[1, 2],
    #   bounds[1, 0], bounds[1, 1], bounds[0, 2],
    #   bounds[1, 0], bounds[1, 1], bounds[1, 2]]
    
    vertices = [
      bounds[0][0], bounds[0][1], bounds[0][2],
      bounds[0][0], bounds[0][1], bounds[1][2],
      bounds[0][0], bounds[1][1], bounds[0][2],
      bounds[0][0], bounds[1][1], bounds[1][2],
      bounds[1][0], bounds[0][1], bounds[0][2],
      bounds[1][0], bounds[0][1], bounds[1][2],
      bounds[1][0], bounds[1][1], bounds[0][2],
      bounds[1][0], bounds[1][1], bounds[1][2]]

    vertices = np.array(vertices,
                        order='C',
                        dtype=np.float64).reshape((-1, 3))
    # Do we need this?
    # vertices -= 0.5

    # if extents is not None:
    #     extents = np.asanyarray(extents, dtype=np.float64)
    #     if extents.shape != (3,):
    #         raise ValueError('Extents must be (3,)!')
    #     vertices *= extents

    # faces = [1, 3, 0, 4, 1, 0, 
    #           0, 3, 2, 2, 4, 0, 1, 7, 3, 5, 1, 4,
    #          5, 7, 1, 3, 7, 2, 6, 4, 2, 2, 7, 6, 6, 5, 4, 7, 5, 6]
    faces = [7, 3, 1, 5, 0, 2, 6, 4, 
              7, 5, 4, 6, 1, 3, 2, 0,
               3, 7, 6, 2, 5, 1, 0, 4]

    faces = np.array(faces,
                     order='C', dtype=np.int64).reshape((-1, 4))

    face_normals = [0, 0, 1, 0, 0, -1,
                      1, 0, 0, -1, 0, 0,
                      0, 1, 0, 0, -1, 0]
    face_normals = np.asanyarray(face_normals,
                                 order='C',
                                 dtype=np.float64).reshape(-1, 3)

    box = ShrinkWrapQuadMesh(vertices=vertices,
                  faces=faces,
                  face_normals=face_normals)

    # return mesh
    return box

  def recalculate_face_normals(self):
    new_face_normals = []
    for fi in range(len(self.faces)):
      f = self.faces[fi]
      fn = np.array([0,0,0])
      if len(f) >= 3:
        v0 = self.vertices[f[0]]
        v1 = self.vertices[f[1]]
        v2 = self.vertices[f[2]]
        fn = ShrinkWrapQuadMesh.get_plane_normal(v0, v1, v2)
      new_face_normals.append(fn)

    self.face_normals = new_face_normals


  def recalculate_vertex_normals(self):
    vertices = self.vertices
    faces = self.faces
    face_normals = self.face_normals
    
    self.vertex_faces = [[] for x in range(len(vertices))]
    for fi in range(len(faces)):
      for vi in faces[fi]:
        self.vertex_faces[vi].append(fi)

    vertex_normals = []
    for vi in range(len(vertices)):
      v_faces = np.take(self.face_normals, self.vertex_faces[vi], axis=0)
      vn = np.mean(v_faces, axis=0)
      vn = vn / np.sqrt(np.sum(vn **2))
      vertex_normals.append(vn)

    self.vertex_normals = np.array(vertex_normals)


  def get_edge_key(vi0, vi1):
    return (str(vi0) + "-" + str(vi1)) if vi0 < vi1 else (str(vi1) + "-" + str(vi0))

  def get_plane_normal(p1, p2, p3):
    v1 = p3 - p1
    v2 = p2 - p1
    n = np.cross(v1, v2)
    n = n / np.sqrt(np.sum(n **2))
    return n

  def preview_meshes(meshes):
    scene = pyrender.Scene()
    for m in meshes:
      scene.add(pyrender.Mesh.from_trimesh(m))
    pyrender.Viewer(scene, use_raymond_lighting=True)

  def preview_initial_fit(meshes, points):
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


  def scale_bounds(bounds, scale):
    center = [(bounds[0][0] + bounds[1][0]) / 2,
      (bounds[0][1] + bounds[1][1]) / 2,
      (bounds[0][2] + bounds[1][2]) / 2]

    extent = [(bounds[1][0] - bounds[0][0]) / 2 * scale,
      (bounds[1][0] - bounds[0][0]) / 2 * scale,
      (bounds[1][0] - bounds[0][0]) / 2 * scale]

    return np.array([
      [center[0] - extent[0], center[1] - extent[1], center[2] - extent[2]],
      [center[0] + extent[0], center[1] + extent[1], center[2] + extent[2]],
    ])

  def subdivide(self):
    # for each faces, add new vertices and reset faces
    # keep track of new vertices index
    # calculate face normal

    edge_mid_points = {}
    edge_mid_point_indices = {}
    # divide all line segs

    new_faces = []
    new_face_normals = []
    new_vertices = []
    new_vertex_indices = []
    next_vertex_index = len(self.vertices)
    new_vertex_indices = []

    quad_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    for face in self.faces:
      
      quad_mid_points = []
      quad_mid_point_indices = []
      
      # Find midpoints for all the edges of the face
      for quad_edge in quad_edges: 
        vi0 = face[quad_edge[0]]
        vi1 = face[quad_edge[1]]
        edge_key = ShrinkWrapQuadMesh.get_edge_key(vi0, vi1)
        
        if edge_key not in edge_mid_points:
          v0 = self.vertices[vi0]
          v1 = self.vertices[vi1]
          mid_point = (v0 + v1) * 0.5
          edge_mid_points[edge_key] = mid_point
          edge_mid_point_indices[edge_key] = next_vertex_index
          new_vertex_indices.append(next_vertex_index)
          
          next_vertex_index += 1
          new_vertices.append(mid_point)

        quad_mid_points.append(edge_mid_points[edge_key])
        quad_mid_point_indices.append(edge_mid_point_indices[edge_key])

      # Find the center point of the original quad_center
      quad_center = (self.vertices[face[0]] + self.vertices[face[1]] + self.vertices[face[2]] + self.vertices[face[3]]) / 4.0
      quad_center_index = next_vertex_index
      new_vertices.append(quad_center)
      new_vertex_indices.append(quad_center_index)
      next_vertex_index += 1
      # Add it as a new vertex

      # construct 4 sub faces
      new_faces.append([face[0], quad_mid_point_indices[0], quad_center_index, quad_mid_point_indices[3]])
      new_faces.append([face[1], quad_mid_point_indices[1], quad_center_index, quad_mid_point_indices[0]])
      new_faces.append([face[2], quad_mid_point_indices[2], quad_center_index, quad_mid_point_indices[1]])
      new_faces.append([face[3], quad_mid_point_indices[3], quad_center_index, quad_mid_point_indices[2]])

      new_face_normals.append(ShrinkWrapQuadMesh.get_plane_normal(quad_center, quad_mid_points[0], quad_mid_points[3]))
      new_face_normals.append(ShrinkWrapQuadMesh.get_plane_normal(quad_center, quad_mid_points[1], quad_mid_points[0]))
      new_face_normals.append(ShrinkWrapQuadMesh.get_plane_normal(quad_center, quad_mid_points[2], quad_mid_points[1]))
      new_face_normals.append(ShrinkWrapQuadMesh.get_plane_normal(quad_center, quad_mid_points[3], quad_mid_points[2]))

    self.vertices = np.concatenate((self.vertices, new_vertices))
    self.faces = new_faces
    self.face_normals = new_face_normals
    self.recalculate_vertex_normals()

    return new_vertex_indices

  def get_min_projection_distances_to_surfaces(self, vertex_indices, surfaces, prioritize_positive_ray=False):
    # tmesh = self.get_tri_mesh()
    vertex_normals = self.vertex_normals
    ray_origins = np.take(self.vertices, vertex_indices, axis=0)
    
    ray_directions_negative = -np.take(vertex_normals, vertex_indices, axis=0)
    ray_directions_positive = np.take(vertex_normals, vertex_indices, axis=0)

    distances = np.repeat(np.inf, len(vertex_indices))
    projection = np.repeat(np.inf, len(vertex_indices))
    location_for_rays = ray_origins.copy()

    for surface in surfaces:
      for direction in [1, -1]:

        ray_directions = ray_directions_positive if direction > 0 else ray_directions_negative

        locations, index_ray, index_tri = surface.ray.intersects_location(
          ray_origins=ray_origins,
          ray_directions=ray_directions,
          multiple_hits=False)

        for l in range(len(locations)):
          ray_index = index_ray[l]
          origin = ray_origins[ray_index]
          dist = np.sqrt(np.sum((locations[l] - origin) ** 2))

          if dist < distances[ray_index]:
            if prioritize_positive_ray and direction == -1 and distances[ray_index] < np.inf:
              # already found the positive ray
              pass
            else:
              distances[ray_index] = dist
              projection[ray_index] = dist * direction
              location_for_rays[ray_index] = locations[l]

    # if inverted:
    #   distances = -distances
    # print('distances', distances)
    # print('projection', projection)
    # print(location_for_rays)

    return projection, location_for_rays
    # Find localtion and dist

  def project_vertices(self, vertex_indices, distances):
    # tmesh = self.get_tri_mesh()
    # vertex_normals = tmesh.vertex_normals
    vertex_normals = self.vertex_normals
    for i in range(len(vertex_indices)):
      idx = vertex_indices[i]
      n = vertex_normals[idx]
      self.vertices[idx] = self.vertices[idx] + (distances[i] * n)

  def get_tri_mesh(self, **kwargs):
    # self.faces
    # return the mesh
    
    tri_faces = trimesh.geometry.triangulate_quads(self.faces)
    box = trimesh.Trimesh(vertices=self.vertices,
                  faces=tri_faces,
                  # face_normals=face_normals,
                  process=False,
                  **kwargs)
    return box

  def create_initial_fit_mesh(surfaces, debug=False):
    # for each corner point of a cube
    #  find normal
    #  filter the surface
    #  for each surface s of the filtered_surfaces
    #   find vertices with the closest normal to the one we're looking
    # return the six vertices

    quad_mesh = ShrinkWrapQuadMesh.from_box(bounds=[[-1, -1, -1],[1,1,1]])

    for vi in range(len(quad_mesh.vertices)):
      vn = quad_mesh.vertex_normals[vi]
      print('finding position for vn ', vn)
      print(np.sqrt(np.sum(vn ** 2)))
      max_dp = -1
      
      selected_surface = None

      for s in surfaces:
        if np.dot(vn, s["direction"]) >= 0:
          print("looking at surface ", s["direction"])
          sm = s["mesh"]
          for svi in range(len(sm.vertices)):
            svn = sm.vertex_normals[svi]
            svn = svn / np.sqrt(np.sum(svn **2))
            #if np.sqrt(np.sum(svn ** 2)) > 1:
            #  print('WARN: normal has magnitude more than 1', np.sqrt(np.sum(svn ** 2)))

            dp = np.dot(vn, svn)
            # print('dot product', dp)

            if dp > max_dp:
              max_dp = dp
              # print('found new max_dp', dp)
              quad_mesh.vertices[vi] = sm.vertices[svi]
              selected_surface = sm

      if(debug):
        ShrinkWrapQuadMesh.preview_meshes([quad_mesh.get_tri_mesh(), selected_surface])

    # quad_mesh.recalculate_face_normals()
    # quad_mesh.recalculate_vertex_normals()

    return quad_mesh

  def vectorize(trimesh, bounding_box, num_subdivisions=5, resolution_multiplier=1, mesh_transform=None, return_result_mesh=False, debug=False):

    if type(trimesh) == list:
      mesh = []
      for tm in trimesh:
        mesh.append(pyrender.Mesh.from_trimesh(tm))
      mesh_bounds = find_common_bounds(trimesh)    
    else:
      mesh = pyrender.Mesh.from_trimesh(trimesh)
      mesh_bounds = trimesh.bounds
      trimesh = [trimesh]


    map_top, s_top = build_depth_surface(mesh, bounding_box, direction='+z', znear=0.001, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
    map_bottom, s_bottom = build_depth_surface(mesh, bounding_box, direction='-z', znear=0.001, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
    map_front, s_front = build_depth_surface(mesh, bounding_box, direction='-y', znear=0.001, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
    map_back, s_back = build_depth_surface(mesh, bounding_box, direction='+y', znear=0.001, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
    map_left, s_left = build_depth_surface(mesh, bounding_box, direction='+x', znear=0.001, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)
    map_right, s_right = build_depth_surface(mesh, bounding_box, direction='-x', znear=0.001, resolution_multiplier=resolution_multiplier, mesh_transform=mesh_transform)

    depth_surfaces = [s_top, s_front, s_back, s_left, s_right, s_bottom]

    if debug:
      for dmap in [map_top, map_bottom, map_front, map_back, map_left, map_right]:
        plt.imshow(dmap, cmap=plt.cm.gray_r)
        plt.show()

      #for s in [s_top, s_front, s_back, s_left, s_right, s_bottom]:
      #  ShrinkWrapQuadMesh.preview_meshes([s])

      ShrinkWrapQuadMesh.preview_meshes(depth_surfaces)

    # quad_mesh = ShrinkWrapQuadMesh.create_initial_fit_mesh([
    #   {'direction': np.array([0,0,1]), 'mesh': s_top},
    #   {'direction': np.array([0,0,-1]), 'mesh': s_bottom},
    #   {'direction': np.array([0,-1,0]), 'mesh': s_front},
    #   {'direction': np.array([0,1,0]), 'mesh': s_back},
    #   {'direction': np.array([1,0,0]), 'mesh': s_left},
    #   {'direction': np.array([-1,0,0]), 'mesh': s_right}
    # ], debug=True)
    
    # if debug:
    #   print('Initial fit')
    #   print(quad_mesh.vertices)
    #   ShrinkWrapQuadMesh.preview_meshes([trimesh, quad_mesh.get_tri_mesh()])

    quad_mesh = ShrinkWrapQuadMesh.from_box(bounds=mesh_bounds)
    indices = range(len(quad_mesh.vertices))
    # distances, points = quad_mesh.get_min_projection_distances_to_surfaces(indices, depth_surfaces)
    distances, points = quad_mesh.get_min_projection_distances_to_surfaces(indices, trimesh)
    quad_mesh.project_vertices(indices, distances)

    if debug:
      ShrinkWrapQuadMesh.preview_initial_fit(trimesh, quad_mesh.vertices)

    # collect initial mesh as vector
    # print(quad_mesh.vertices.flatten())
    result_vector = quad_mesh.vertices.flatten()

    for i in range(num_subdivisions):
      new_vertices = quad_mesh.subdivide()
      d, p = quad_mesh.get_min_projection_distances_to_surfaces(new_vertices, depth_surfaces, prioritize_positive_ray=False)
      quad_mesh.project_vertices(new_vertices, d)

      result_vector = np.concatenate((result_vector, d))

      if debug:
        print('n_subdivisions', i)
        ShrinkWrapQuadMesh.preview_meshes([quad_mesh.get_tri_mesh()])
    
    if return_result_mesh:
      return result_vector, quad_mesh
    else:
      return result_vector

  def devectorize(vector, debug=False, max_subdiv=10):
    quad_mesh = ShrinkWrapQuadMesh.from_box(bounds=[[-1,-1,-1],[1,1,1]])
    quad_mesh.vertices = vector[:24].reshape(-1, 3)
    vector = vector[24:]
    quad_mesh.recalculate_face_normals()
    quad_mesh.recalculate_vertex_normals()

    if debug:
      ShrinkWrapQuadMesh.preview_meshes([quad_mesh.get_tri_mesh()])

    num_subdivisions = 0
    while num_subdivisions < max_subdiv:
      
      num_new_vertices = len(quad_mesh.faces) * 4
      if len(vector) < num_new_vertices:
        # No more vectors to use
        break

      new_vertices = quad_mesh.subdivide()
      d = vector[:len(new_vertices)]
      quad_mesh.project_vertices(new_vertices, d)
      vector = vector[len(new_vertices):]

      if debug:
        print('n_subdivisions', num_subdivisions)
        ShrinkWrapQuadMesh.preview_meshes([quad_mesh.get_tri_mesh()])

      num_subdivisions += 1

    return quad_mesh

# print(trimesh.geometry.triangulate_quads(np.array([[1,2,3,4]])))


