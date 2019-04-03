import os
import trimesh

# Root dir
# model_dir = '../models/02876657/' # bottles
# model_dir = '../models/02924116/' # bus

model_dir = '../models/03938244/' # pillow



epsilon = 0.0001
common_bounds = [[-epsilon, -epsilon, -epsilon], [epsilon, epsilon, epsilon]]
input_models = []
rejected_models_count = 0

# [[-0.380373, -0.198178, -0.458945], [0.380373, 0.198178, 0.458945]]

for filename in os.listdir(model_dir):
  try:
    print(filename)
    target = trimesh.load(model_dir + filename + '/model.obj')
    if type(target) != list:
      input_models.append(filename)
      print(target.bounds)
      for axis in range(3):
        if target.bounds[0][axis] < common_bounds[0][axis]:
          common_bounds[0][axis] = target.bounds[0][axis]
        if target.bounds[1][axis] > common_bounds[1][axis]:
          common_bounds[1][axis] = target.bounds[1][axis]
    else:
      rejected_models_count +=1

    # TODO: deal with list mesh
  except FileNotFoundError:
    # print(err)
    print(error)

print('model_count', len(input_models))
print('rejected_models_count', rejected_models_count)
print('common_bounds', common_bounds)