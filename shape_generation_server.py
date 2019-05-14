import os
import io
import trimesh
import pandas as pd
import tensorflow as tf
from shrink_wrap_quad_mesh import *
# from network import create_layers
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import fully_connected
from flask import Flask, request
from flask import send_file
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

def preview_meshes(meshes):
  scene = pyrender.Scene()
  for m in meshes:
    scene.add(pyrender.Mesh.from_trimesh(m))
  pyrender.Viewer(scene, use_raymond_lighting=True)

# Encode
# saver = tf.train.Saver()

# Init
sess = tf.Session()
model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], './tmp/saved_model_washingmachine')
loaded_graph = tf.get_default_graph()

input_tensor_name = model.signature_def['decode'].inputs['code'].name
input_tensor = loaded_graph.get_tensor_by_name(input_tensor_name)
output_tensor_name = model.signature_def['decode'].outputs['output_vector'].name
output_tensor = loaded_graph.get_tensor_by_name(output_tensor_name)
# vec = output_tensor.eval(feed_dict={input_tensor: np.random.rand(100,10)}, session=sess)
# df = pd.DataFrame(vec)
# df.to_csv('./data/results_random_3.csv')

# for vector in vec:
#   output_mesh = ShrinkWrapQuadMesh.devectorize(vector, debug=False)
#   preview_meshes([output_mesh.get_tri_mesh()])

@app.route("/", methods=['POST'])
def generate_random():
  
  # TODO Get vec from client
  input_feature_vec = np.random.rand(1,10)
  if request.json["input"] is not None:
    input_feature_vec = np.array([request.json["input"]])

  vec = output_tensor.eval(feed_dict={input_tensor: input_feature_vec}, session=sess)
  output_mesh = ShrinkWrapQuadMesh.devectorize(vec[0], debug=False)
  # preview_meshes([output_mesh.get_tri_mesh()])
  
  scene = pyrender.Scene()
  
  mesh = pyrender.Mesh.from_trimesh(output_mesh.get_tri_mesh())
  scene.add(mesh)
  
  camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
  # s = np.sqrt(2)/2
  # camera_pose = np.array([
  #   [0.0, -s,   s,   0.3],
  #   [1.0,  0.0, 0.0, 0.0],
  #   [0.0,  s,   s,   10.0],
  #   [0.0,  0.0, 0.0, 1.0],
  # ])
  camera_pose = np.array([
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  1.0, 0.0, 0.0],
    [0.0,  0.0, 1.0, 20.0],
    [0.0,  0.0, 0.0, 1.0],
  ])
  scene.add(camera, pose=camera_pose)

  light = pyrender.SpotLight(color=np.ones(3), intensity=20.0,
                              innerConeAngle=np.pi/16.0,
                              outerConeAngle=np.pi/2.0)
  scene.add(light, pose=camera_pose)
  
  r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
  color, depth = r.render(scene)
  # print(color)

  pil_img = Image.fromarray(color)
  buff = io.BytesIO()
  pil_img.save(buff, format="JPEG")
  # return the buff
  buff.seek(0)
  return send_file(buff, mimetype='image/jpeg')

  # return 'hello'



  
