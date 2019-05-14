import os
import trimesh
import pandas as pd
from shrink_wrap_quad_mesh import *

# Root dir
# model_dir = '../models/02876657/' # bottles
# model_dir = '../models/03467517/' # guitars
# model_dir = '../models/03938244/' # pillow

# model_dir = '../models/03513137/' # helmet

# model_dir = '../models/birdhouse/'
# model_dir = '../models/washingmachine/'
model_dir = '../models/car_temp/'

# model_dir = '../models/03761084/' # microwave
# model_dir = '../models/02924116/' # bus


epsilon = 0.0001
# common_bounds = [[-epsilon, -epsilon, -epsilon], [epsilon, epsilon, epsilon]]
input_models = []
rejected_models_count = 0
resolution_multiplier = 300
max_models = 1000
preview_result_mesh = True
preview_devectorization = False

# common_bounds = [[-0.380373, -0.198178, -0.458945], [0.380373, 0.198178, 0.458945]]

# common_bounds = [[-0.5, -0.6, -0.5], [0.5, 0.6, 0.5]] 

# common_bounds = [[-0.5, -0.3, -0.5], [0.5, 0.3, 0.5]] # pillow

# common_bounds = [[-0.5, -0.4, -0.5], [0.5, 0.4, 0.5]] # helmet

# common_bounds = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]] # birdhouse

# common_bounds = [[-0.5, -0.4, -0.5], [0.5, 0.4, 0.5]] # printer

# common_bounds = [[-0.4, -0.5, -0.5], [0.4, 0.5, 0.5]] # washingmachine

common_bounds = [[-0.25, -0.25, -0.5], [0.25, 0.25, 0.5]] # car_temp

# common_bounds = [[-0.608658, -0.651665, -0.636719], [0.608658, 0.651665, 0.636719]] # microwave

# common_bounds = [[-0.695646, -0.493881, -0.433217], [0.695646, 0.493881, 0.433217]] # bus

# [[-0.0721975, -0.492051, -0.249519], [0.0721975, 0.492051, 0.249519]]

# [[-0.325698, -0.481754, -0.30431], [0.325698, 0.481754, 0.30431]]

def preview_meshes(meshes):
  scene = pyrender.Scene()
  for m in meshes:
    scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

with open('./data/vectors_car.csv', 'a') as f:
  # vectors = []
  files = os.listdir(model_dir)
  num_processed = 0
  num_all_files = len(files)
  for filename in files:
    
    print('Processing ', num_processed, '/', num_all_files)
    if len(input_models) >= max_models:
      continue

    try:

      all_meshes = trimesh.load(model_dir + filename + '/model.obj')
      if type(all_meshes) != list:
        all_meshes = [all_meshes] 
      
      vector, qmesh = ShrinkWrapQuadMesh.vectorize(all_meshes, 
                common_bounds, 
                num_subdivisions=5, 
                resolution_multiplier=resolution_multiplier,
                return_result_mesh=True,
                debug=preview_result_mesh)

      if preview_result_mesh:
        preview_meshes(all_meshes)
        preview_meshes([qmesh.get_tri_mesh()])

      print('shape', vector.shape)
      # vectors.append(vector)
      df = pd.DataFrame([vector])
      df.to_csv(f, header= (num_processed == 0))
      input_models.append(filename)
      num_processed += 1
      print()

      if preview_devectorization:
        output_mesh = ShrinkWrapQuadMesh.devectorize(vector, debug=False)
        preview_meshes([output_mesh.get_tri_mesh()] + all_meshes)

      # TODO: deal with list mesh
    except FileNotFoundError:
      # print(err)
      pass

    except Exception as e:
      print(e)
      rejected_models_count +=1
      pass

  print('model_count', len(input_models))
  print('rejected_models_count', rejected_models_count)
  # print('common_bounds', common_bounds)

  # df = pd.DataFrame(vectors)
  # df.to_csv('./data/vectors_helmet.csv')