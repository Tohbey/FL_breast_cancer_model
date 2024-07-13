import flwr as fl
import tensorflow as tf
from tensorflow import keras
import os
from os import listdir
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from shutil import copyfile

import glob
import PIL
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image
from PIL import ImageEnhance
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from torchvision import transforms
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
import shutil

# multi-cancer datasets
breast_cancer_path = "/kaggle/input/multi-cancer/Multi Cancer/Breast Cancer/"
benign_dir = os.path.join(breast_cancer_path, 'breast_benign')
malignant_dir = os.path.join(breast_cancer_path, 'breast_malignant')

train_dir = '/kaggle/breast_cancer/training_data'
val_dir = '/kaggle/breast_cancer/validation_data'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

def split_copy(class_dir, train_output_dir, val_output_dir):
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    for img in train_images:
        copyfile(img, os.path.join(train_output_dir, os.path.basename(img)))

    for img in val_images:
        copyfile(img, os.path.join(val_output_dir, os.path.basename(img)))

split_copy(benign_dir, os.path.join(train_dir, 'breast_benign'), os.path.join(val_dir, 'breast_benign'))
split_copy(malignant_dir, os.path.join(train_dir, 'breast_malignant'), os.path.join(val_dir, 'breast_malignant'))

# histopathology images
breast_imgs = glob.glob('/kaggle/input/breast-histopathology-images/*/*/*')

imagePatches = [breast_imgs[i] for i in range(len(breast_imgs)) if 'IDC' not in breast_imgs[i]]

idc_dir = os.path.join(breast_cancer_path, 'idc')
non_idc_dir = os.path.join(breast_cancer_path, 'non_idc')

for img in imagePatches:
    if img[-5] == '0' :
        shutil.move(img, os.path.join(non_idc_dir, os.path.basename(img)))
    
    elif img[-5] == '1' :
        shutil.move(img, os.path.join(idc_dir, os.path.basename(img)))
        
# split histopathology images
split_copy(idc_dir, os.path.join(train_dir, 'idc'), os.path.join(val_dir, 'idc'))
split_copy(non_idc_dir, os.path.join(train_dir, 'non_idc'), os.path.join(val_dir, 'non_idc'))

     
def load_data():
    train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
    '/kaggle/breast_cancer/training_data',
    target_size=(150, 150),
    batch_size=512,
    class_mode='binary'
    )
    
    validation_generator = val_datagen.flow_from_directory(
    '/kaggle/breast_cancer/validation_data',
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary'
    )

# Define the model
def get_image_model(input_shape, num_classes):
    # Load the VGG16 model pre-trained on ImageNet, without the top fully connected layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model (we don't want to update the weights of the pre-trained layers)
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers on top of the base model
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Flower client
class FlowerClient1(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=1, batch_size=32, verbose=0)
        return self.get_parameters(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}

# Load data
x_train, y_train, x_test, y_test = load_data()

# Create model
model = get_image_model()

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient1(model, x_train, y_train, x_test, y_test))
