import os
import trimesh
from shrink_wrap_quad_mesh import *

# Root dir
# model_dir = '../models/02876657/' # bottles
# model_dir = '../models/03467517/' # guitars
model_dir = '../models/03938244/' # pillow

epsilon = 0.0001
# common_bounds = [[-epsilon, -epsilon, -epsilon], [epsilon, epsilon, epsilon]]
input_models = []
rejected_models_count = 0
resolution_multiplier = 300
max_models = 5
preview_result_mesh = True
preview_devectorization = False

# common_bounds = [[-0.380373, -0.198178, -0.458945], [0.380373, 0.198178, 0.458945]]

# common_bounds = [[-0.5, -0.6, -0.5], [0.5, 0.6, 0.5]] 

common_bounds = [[-0.5, -0.3, -0.5], [0.5, 0.3, 0.5]] # pillow

# [[-0.0721975, -0.492051, -0.249519], [0.0721975, 0.492051, 0.249519]]

# [[-0.325698, -0.481754, -0.30431], [0.325698, 0.481754, 0.30431]]

def preview_meshes(meshes):
  scene = pyrender.Scene()
  for m in meshes:
    scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

for filename in os.listdir(model_dir):
  
  if len(input_models) >= max_models:
    continue

  try:

    target = trimesh.load(model_dir + filename + '/model.obj')
    if type(target) != list:
      input_models.append(filename)

      vector, qmesh = ShrinkWrapQuadMesh.vectorize(target, 
                common_bounds, 
                num_subdivisions=5, 
                resolution_multiplier=resolution_multiplier,
                return_result_mesh=True,
                debug=True)

      if preview_result_mesh:
        preview_meshes([target])
        preview_meshes([qmesh.get_tri_mesh(), target])

      print(vector.shape)

      if preview_devectorization:
        print('preview_devectorization')
        output_mesh = ShrinkWrapQuadMesh.devectorize(vector, debug=False)
        preview_meshes([output_mesh.get_tri_mesh()])


    else:
      rejected_models_count +=1

    # TODO: deal with list mesh
  except FileNotFoundError:
    # print(err)
    pass

  except ValueError:
    pass

  except Exception as e:
    print(e)
    pass

print('model_count', len(input_models))
print('rejected_models_count', rejected_models_count)
print('common_bounds', common_bounds)