import os
import trimesh
import pandas as pd
import tensorflow as tf
from shrink_wrap_quad_mesh import *
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected

def preview_meshes(meshes):
  scene = pyrender.Scene()
  for m in meshes:
    scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

# Encode
# saver = tf.train.Saver()

with tf.Session(graph=tf.Graph()) as sess:
  model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './tmp/saved_model')
  loaded_graph = tf.get_default_graph()

  input_tensor_name = model.signature_def['decode'].inputs['code'].name
  input_tensor = loaded_graph.get_tensor_by_name(input_tensor_name)
  output_tensor_name = model.signature_def['decode'].outputs['output_vector'].name
  output_tensor = loaded_graph.get_tensor_by_name(output_tensor_name)
  
  vec = output_tensor.eval(feed_dict={input_tensor: np.random.rand(100,10)}, session=sess)
  df = pd.DataFrame(vec)
  df.to_csv('./data/results_random_3.csv')

  for vector in vec:
    output_mesh = ShrinkWrapQuadMesh.devectorize(vector, debug=False)
    preview_meshes([output_mesh.get_tri_mesh()])
  
