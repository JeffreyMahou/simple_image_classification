import os
import PIL
import matplotlib.pyplot as plt
import numpy as np

# load image
# path = PATH
name = "Lenna.png"
img = PIL.Image.open(os.path.join(path, name))

# Afficher l'image chargée
img.show()

##

img = np.array(img)

# histogram of the image
n, bins, patches = plt.hist(img.flatten(), bins=range(256))
plt.show()


## CNN

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense

my_VGG16 = Sequential()  # Création d'un réseau de neurones vide

# first conv layer with relu activation
my_VGG16.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))

# second conv layer with relu activation
my_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

# first pooling layer
my_VGG16.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

my_VGG16.add(Flatten())  # Conversion of 3d matrices into 1d vectors

# first fully connected layer with relu
my_VGG16.add(Dense(4096, activation='relu'))

# second fully connected layer with relu
my_VGG16.add(Dense(4096, activation='relu'))

# last fully connected layer with softmax to determine classes
my_VGG16.add(Dense(1000, activation='softmax'))


##

import os
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

from keras.applications.vgg16 import VGG16

model = VGG16() # Création du modèle VGG-16 implementé par Keras

# path = PATH
name = "cat.jpg"

img = load_img(os.path.join(path, name), target_size=(224, 224))  # load the image
img = img_to_array(img)  # convert into numpy array
img = preprocess_input(img)  # pretreatment as VGG wants it
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))  # Create a collection of images


y = model.predict(img)  # Predict the right class

##


from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras import Model
# Charger VGG-16 pré-entraîné sur ImageNet et sans les couches fully-connected
# load VGG model pretrained on ImageNet and remove top layers for different outputs
model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# output of pretrained model
x = model.output

# new output with 10 classes
predictions = Dense(10, activation='softmax')(x)

# new model
new_model = Model(inputs=model.input, outputs=predictions)

#first strategy
for layer in model.layers:
   layer.trainable = True

#second strategy
for layer in model.layers:
   layer.trainable = False

#third strategy
# don't train on the 5 lowest layers
for layer in model.layers[:5]:
   layer.trainable = False

# compile
new_model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

# # train on some data
# model_info = new_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)








