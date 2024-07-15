import flwr as fl
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Concatenate
from keras.optimizers import Adam

from utils import countingFilesInDirectory, plotClientData, split_copy

results_list = []

# multi-cancer datasets
breast_cancer_path = "/kaggle/input/multi-cancer/Multi Cancer/Breast Cancer/"
benign_dir = os.path.join(breast_cancer_path, 'breast_benign')
malignant_dir = os.path.join(breast_cancer_path, 'breast_malignant')

train_dir = '/kaggle/breast_cancer/training_data'
val_dir = '/kaggle/breast_cancer/validation_data'

histopathology_path = "/kaggle/histopathology"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Split multi-cancer images
split_copy(benign_dir, os.path.join(train_dir, 'breast_benign'), os.path.join(val_dir, 'breast_benign'))
split_copy(malignant_dir, os.path.join(train_dir, 'breast_malignant'), os.path.join(val_dir, 'breast_malignant'))

# histopathology images
input_path = '/kaggle/input/breast-histopathology-images'

idc_dir = os.path.join(histopathology_path, 'idc')
non_idc_dir = os.path.join(histopathology_path, 'non_idc')

# Create directories if they do not exist
os.makedirs(idc_dir, exist_ok=True)
os.makedirs(non_idc_dir, exist_ok=True)

# Get list of image paths
breast_imgs = glob.glob(os.path.join(input_path, '*/*/*'))

# Filter image patches
imagePatches = [img for img in breast_imgs if 'IDC' not in img]

def copy_image(img):
    if img[-5] == '0':
        shutil.copyfile(img, os.path.join(non_idc_dir, os.path.basename(img)))
    elif img[-5] == '1':
        shutil.copyfile(img, os.path.join(idc_dir, os.path.basename(img)))

# Use ThreadPoolExecutor to parallelize copying files
with ThreadPoolExecutor() as executor:
    executor.map(copy_image, imagePatches)

# Split histopathology images
split_copy(idc_dir, os.path.join(train_dir, 'idc'), os.path.join(val_dir, 'idc'))
split_copy(non_idc_dir, os.path.join(train_dir, 'non_idc'), os.path.join(val_dir, 'non_idc'))

print(countingFilesInDirectory(train_dir))
print(countingFilesInDirectory(val_dir))

# Load CSV data
def load_data_csv():
    # Load the dataset
    wisconsin_df = pd.read_csv("/kaggle/input/breast-cancer/data.csv")
    
    # Drop unnecessary columns
    wisconsin_df.drop(columns=['Unnamed: 32', 'id'], inplace=True)
    
    # Encode the 'diagnosis' column
    wisconsin_df['diagnosis'] = LabelEncoder().fit_transform(wisconsin_df['diagnosis'])
    
    # Separate features and labels
    features = wisconsin_df.drop(columns=['diagnosis'])
    labels = wisconsin_df['diagnosis']
    
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Convert to numpy arrays
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()
    
    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes=2)
    test_labels = to_categorical(test_labels, num_classes=2)
    
    return train_data, test_data, train_labels, test_labels

# Load image data
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
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    return train_generator, validation_generator

# Define the combined model
def get_combined_model(image_shape=(224, 224, 3), num_csv_features=30, num_classes=2):
    # Image model (VGG16)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=image_shape)
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    image_output = Dense(128, activation='relu')(x)
    
    # CSV model
    csv_input = Input(shape=(num_csv_features,))
    y = Dense(64, activation='relu')(csv_input)
    y = Dense(32, activation='relu')(y)
    csv_output = Dense(128, activation='relu')(y)
    
    # Combine models
    combined = Concatenate()([image_output, csv_output])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.5)(z)
    z = Dense(num_classes, activation='softmax')(z)
    
    model = Model(inputs=[base_model.input, csv_input], outputs=z)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_img_gen, val_img_gen, train_csv, train_csv_labels, val_csv, val_csv_labels):
        self.model = model
        self.train_img_gen = train_img_gen
        self.val_img_gen = val_img_gen
        self.train_csv = train_csv
        self.train_csv_labels = train_csv_labels
        self.val_csv = val_csv
        self.val_csv_labels = val_csv_labels

    def get_parameters(self):
        return self.model.get_weights()

    def set_parameters(self, parameters):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        # Update local model parameters
        self.set_parameters(parameters)
        
        # Train the model using hyperparameters from config
        history = self.model.fit(
            [self.train_img_gen.next()[0], self.train_csv], self.train_csv_labels, 
            epochs=1, batch_size=32, verbose=0
        )
        
        # Return updated model parameters and results
        parameters_prime = self.get_parameters()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        print("Local Training Metrics on client 1: {}".format(results))
    
        results_list.append(results)    
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(
            [self.val_img_gen.next()[0], self.val_csv], self.val_csv_labels
        )
        num_examples_test = len(self.val_csv)
        
        print("Evaluation accuracy on Client 1 after weight aggregation : ", accuracy)
        return loss, num_examples_test, {"accuracy": accuracy}

# Load image data
train_generator, val_generator = load_data()

# Load CSV data
train_csv, val_csv, train_csv_labels, val_csv_labels = load_data_csv()

# Create model
model = get_combined_model()

# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", 
    client=FlowerClient(model, train_generator, val_generator, train_csv, train_csv_labels, val_csv, val_csv_labels)
)

plotClientData(results_list)