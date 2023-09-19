# -*- coding: utf-8 -*-
"""
Created on Thu May  7 22:38:15 2020

@author: pavit
"""

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing import image
import numpy as np

from keras.preprocessing.image import img_to_array
from keras.models import load_model
from tensorflow.keras import backend
import os

# Create flask instance
app = Flask(__name__)

#Set Max size of file as 10MB.
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init():
    global graph
    graph = tf.get_default_graph()

# Function to load and prepare the image in right shape
def read_image(filename):
    # Load the image
    img = load_img(filename, color_mode='grayscale', target_size=(100, 100))
    # Convert the image to array
    img = img_to_array(img)
    # Reshape the image into a sample of 1 channel
    img = img.reshape(1, 300, 300, 3)
    # Prepare it as pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img

@app.route("/", methods=['GET', 'POST'])
def home():

    return render_template('home.html')

@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        print("h0")
        print(file)
        try:
            if file and allowed_file(file.filename):
                filename=file.filename
                print(filename)
                print("h1")
                file_path = os.path.join('static/images/', filename)
                print(file_path)
                file.save(file_path)
                img = image.load_img(file_path, target_size=(100, 100))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                #img = read_image(file_path)
                print(x)
                print("h2")
                #'print(img)
                # Predict the class of an image

                with graph.as_default():
                    print("h3")
                    model1 = tf.keras.models.load_model('fruitsv2.h5')
                    class_prediction = model1.predict(x)
                    print(class_prediction)

                #Map apparel category with the numerical class
                if class_prediction[0][0]==1:
                    product="apple"
                elif class_prediction[0][1]==1:
                    product="banana"
                else:
                    product = "lemon"
                return render_template('predict.html', product = product, user_image = file_path)
        except Exception as e:
            return "Unable to read the file. Please check if the file extension is correct."

    return render_template('predict.html')

if __name__ == "__main__":
    init()
    app.run()