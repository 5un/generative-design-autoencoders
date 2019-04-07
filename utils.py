import trimesh

def find_common_bounds(trimeshes):
  epsilon = 0.0001
  common_bounds = [[-epsilon, -epsilon, -epsilon], [epsilon, epsilon, epsilon]]

  for m in trimeshes:
    for axis in range(3):
      if m.bounds[0][axis] < common_bounds[0][axis]:
        common_bounds[0][axis] = m.bounds[0][axis]
      if m.bounds[1][axis] > common_bounds[1][axis]:
        common_bounds[1][axis] = m.bounds[1][axis]

  return common_bounds