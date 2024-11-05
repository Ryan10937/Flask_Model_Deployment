from tensorflow.keras import layers, models, datasets
import numpy as np
from PIL import Image
def define_and_compile_model():
 
  model=models.Sequential(
    [
      layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.MaxPooling2D((2, 2)),
      layers.Conv2D(64, (3, 3), activation='relu'),
      layers.Flatten(),
      layers.Dense(64, activation='relu'),
      layers.Dense(10, activation='softmax')
    ]
  )
  # Compile the model
  model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  return model

def train_model(model,X,y):
  model.fit(X,y,epochs=10)
  return model

def evaluate_model(model,X,y):
  model.evaluate(X,y)
def save_model(model,path):
  model.save(path)
def load_model(path):
  return models.load_model(path)
def infer(model,image_path):
  int_to_class = {
      0: 'airplane',
      1: 'automobile',
      2: 'bird',
      3: 'cat',
      4: 'deer',
      5: 'dog',
      6: 'frog',
      7: 'horse',
      8: 'ship',
      9: 'truck'
    }
  x_size,y_size=32,32
  image = np.expand_dims(np.array(Image.open(image_path).resize((x_size,y_size))), axis=0)
  return int_to_class[np.argmax(model.predict(image)[0])]

def inference_pipeline(image_path):
  # model= define_and_compile_model()
  model = load_model('models/simple_cnn.keras')
  return infer(model,image_path=image_path)

def make_new_cifar10_model():
  # Load the CIFAR-10 dataset
  (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
  x_train, x_test = x_train / 255.0, x_test / 255.0

  model = define_and_compile_model()

  train_model(model,X=x_train,y=y_train)
  
  evaluate_model(model,X=x_test,y=y_test)

  save_model(model,'models/simple_cnn.keras')