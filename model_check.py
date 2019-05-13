import os
import trimesh

# Root dir
# model_dir = '../models/02876657/' # bottles
# model_dir = '../models/02924116/' # bus

model_dir = '../models/03513137/' # helmet

model_dir = '../models/birdhouse/' # helmet

# model_dir = '../models/03938244/' # pillow

# model_dir = '../models/03761084/' # pillow



epsilon = 0.0001
common_bounds = [[-epsilon, -epsilon, -epsilon], [epsilon, epsilon, epsilon]]
input_models = []
rejected_models_count = 0

# [[-0.380373, -0.198178, -0.458945], [0.380373, 0.198178, 0.458945]]

for filename in os.listdir(model_dir):
  try:
    print(filename)
    target = trimesh.load(model_dir + filename + '/model.obj')
    
    all_meshes = target
    if type(target) != list:
      all_meshes = [target]
    
    for m in all_meshes:
      print(m.bounds)
      for axis in range(3):
        if m.bounds[0][axis] < common_bounds[0][axis]:
          common_bounds[0][axis] = m.bounds[0][axis]
        if m.bounds[1][axis] > common_bounds[1][axis]:
          common_bounds[1][axis] = m.bounds[1][axis]

    # TODO: deal with list mesh
  except FileNotFoundError as error:
    # print(err) 
    print(error)
  except ValueError as error:
    print(error)

print('model_count', len(input_models))
print('rejected_models_count', rejected_models_count)
print('common_bounds', common_bounds)