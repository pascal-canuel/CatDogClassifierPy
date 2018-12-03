import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import sys
import cv2
import time
import glob

def predict(filename):
    image = cv2.imread(filename)     
    if(image is not None and image.data):
        image = cv2.resize(image, (64, 64))
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
    class_names = ['cat', 'dog']

    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), input_shape= (64, 64, 3), activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size = (2,2)),
        keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
        keras.layers.MaxPooling2D(pool_size = (2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(units = 128, activation = 'relu'),
        keras.layers.Dense(units = 2, activation = 'sigmoid')])

    model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.load_weights('weights.h5');

    image = (np.expand_dims(image,0))
    prediction = model.predict(image) 
   
    #print('its a ', class_names[np.argmax(prediction[0])])

    print('cat:', prediction[0][0])
    print('dog:', prediction[0][1])

if __name__ == "__main__":
    predict(sys.argv[1])