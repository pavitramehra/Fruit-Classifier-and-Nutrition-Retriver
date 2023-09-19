# -*- coding: utf-8 -*-
"""
Created on Sun May  3 00:57:22 2020

@author: pavit
"""

import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('val_loss')<0.2:
            print("loss is low so stopped training")
            self.model.stop_training=True

TRAINING_DIR = "E:/fruits/training"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "E:/fruits/test"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=20
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=20
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.3),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


model.summary()
print(model.summary())
opt = tf.keras.optimizers.Adam()
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = mycallback()
history = model.fit(train_generator, epochs=16, validation_data = validation_generator, verbose = 1,callbacks=[callback])
#model.save('fruitsv4.h5')
#model = tf.keras.models.load_model('fruitsv1.h5')



#SINGLE IMAGE
'''
from keras.preprocessing import image
import numpy as np

path ="E:/fruits-360/Test/Apple Red 1/3_100"
img = image.load_img(path, target_size=(100, 100))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
images = np.vstack([x])
classes=model.predict(images)
print(classes)
'''
import matplotlib.pyplot as plt


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy optm=rmsprop tsize=100')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss with optm=rmsprop tsize=100')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()




'''
import cv2
import os
import glob
import numpy as np
img_dir = "E:/fruits-360/Test/Kiwi/3_100" # Enter Directory of all images 
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
#img = cv2.imread('C:/Users/pavit/Desktop/fruits-360/Test/Apple Red 3/4_100.jpg')
#.imshow('image',img)
data = []
for f1 in files:
    img = cv2.imread(f1)
    img = cv2.resize(img,(100,100))
    img = np.expand_dims(img, axis=0)
  #  img = np.reshape(img,[1,150,150,3])
    classes=model.predict(img)
    print(classes)
    data.append(img)


'''
'''
#VALIDATION IMAGES
STEP_SIZE_TEST=32

validation_generator.reset()
pred=model.predict_generator(validation_generator,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
labels = (validation_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
print(pred)
print(predictions)
'''




#LIST of IMAGES
'''
folder = "E:/fruits-360/Training/Strawberry/"

import os
lis = sorted(os.listdir(folder))

 #["frame_00", "frame_01", "frame_02", ...]

from PIL import Image 
import numpy as np 
from keras.preprocessing import image
import numpy as np
 # img = cv2.imread(os.path.join(folder,i))
    #img = cv2.resize(img,(150,150))
video_array = []
for i in lis:
    img = image.load_img(os.path.join(folder,i), target_size=(100, 100))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes=model.predict(images)
    print(classes)
    video_array.append(np.asarray(images)) #.transpose(1, 0, 2))

video_array = np.array(video_array)
print(video_array.shape)
#(75, 50, 100, 3)
'''