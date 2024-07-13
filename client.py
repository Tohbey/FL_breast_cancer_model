import flwr as fl
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import numpy as np
import shutil
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Concatenate
from keras.optimizers import Adam

# Directories for image data
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
        shutil.copyfile(img, os.path.join(train_output_dir, os.path.basename(img)))
    for img in val_images:
        shutil.copyfile(img, os.path.join(val_output_dir, os.path.basename(img)))

split_copy(benign_dir, os.path.join(train_dir, 'breast_benign'), os.path.join(val_dir, 'breast_benign'))
split_copy(malignant_dir, os.path.join(train_dir, 'breast_malignant'), os.path.join(val_dir, 'breast_malignant'))

# Load CSV data
def load_data_csv():
    wisconsin_df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/Breast Cancer Wisconsin.csv")
    wisconsin_df.drop(columns=['id', 'Unnamed: 32'], inplace=True)
    wisconsin_df['diagnosis'] = LabelEncoder().fit_transform(wisconsin_df['diagnosis'])
    features = wisconsin_df.drop(columns=['diagnosis'])
    labels = wisconsin_df['diagnosis']
    
    train_data, test_data, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)

    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()
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
        self.set_parameters(parameters)
        self.model.fit(
            [self.train_img_gen.next()[0], self.train_csv], self.train_csv_labels, 
            epochs=1, batch_size=32, verbose=0
        )
        return self.get_parameters(), len(self.train_csv), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(
            [self.val_img_gen.next()[0], self.val_csv], self.val_csv_labels
        )
        return loss, len(self.val_csv), {"accuracy": accuracy}

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
