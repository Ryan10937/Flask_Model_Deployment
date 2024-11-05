from flask import Flask,jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

from scripts.model_funcs import make_new_cifar10_model,inference_pipeline

app = Flask(__name__)

@app.route('/')
def hello_world():
  return "Welcome to my 'deployed' model page :D"

@app.route('/train_model')
def train_model():
  print('Training Model')
  make_new_cifar10_model()
  return "Finished Training"

@app.route('/infer/<path:image_path>')
def infer_on_image(image_path):
  return jsonify({"result": str(inference_pipeline(image_path))})


make_new_cifar10_model
if __name__ == '__main__':
  print('Starting App')

  app.run()
  # image_path='data/image.jpg'
  # print(inference_pipeline(image_path))
  
  # make_new_cifar10_model()