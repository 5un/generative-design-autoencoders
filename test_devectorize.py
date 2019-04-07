import os
import trimesh
import pandas as pd
from shrink_wrap_quad_mesh import *

def preview_meshes(meshes):
  scene = pyrender.Scene()
  for m in meshes:
    scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)
  

df = pd.read_csv('./data/out_2.csv')
# df = pd.read_csv('./data/vectors_pillow.csv')
print(df.shape)

df = df.iloc[:,1:]

vectors = df.values

print(vectors.shape)

for vector in vectors:
  output_mesh = ShrinkWrapQuadMesh.devectorize(vector, debug=False)
  preview_meshes([output_mesh.get_tri_mesh()])