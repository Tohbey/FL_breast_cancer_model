import flwr as fl
import os
from os import listdir
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import seaborn as sns
import glob
import shutil
import PIL
import random
import tensorflow as tensorflow
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image
from PIL import ImageEnhance
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from torchvision import transforms
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, Input, Concatenate
from keras.optimizers import Adam
from utils import countingFilesInDirectory, plotClientData, split_copy

results_list = []


# DDSM_Mammography_Datasets
mammography_breast_cancer_path = '/kaggle/input/ddsm-mammography'

train_dir = '/kaggle/breast_cancer/client2/training_data'
val_dir = '/kaggle/breast_cancer/client2/validation_data'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

mammography_path = '/kaggle/work/breast-mammography-images'
negative_path = os.path.join(mammography_path, 'negative')
benign_classification_path = os.path.join(mammography_path, 'benign_classification')
benign_mass_path = os.path.join(mammography_path, 'benign_mass')
malignant_classification_path = os.path.join(mammography_path, 'malignant_classification')
malignant_mass_path = os.path.join(mammography_path, 'malignant_mass')

os.makedirs(negative_path, exist_ok=True)
os.makedirs(benign_classification_path, exist_ok=True)
os.makedirs(benign_mass_path, exist_ok=True)
os.makedirs(malignant_classification_path, exist_ok=True)
os.makedirs(malignant_mass_path, exist_ok=True)

feature_dictionary = {
    'label': tensorflow.io.FixedLenFeature([], tensorflow.int64),
    'label_normal': tensorflow.io.FixedLenFeature([], tensorflow.int64),
    'image': tensorflow.io.FixedLenFeature([], tensorflow.string)
}

Image_height = 224
Image_width = 224

def _parse_function(example):
    parsed_example = tensorflow.io.parse_single_example(example, feature_dictionary)
    return parsed_example

def read_data(filename):
    full_dataset = tensorflow.data.TFRecordDataset(filename,num_parallel_reads=tensorflow.data.experimental.AUTOTUNE)
    print(full_dataset)
    full_dataset = full_dataset.shuffle(buffer_size=31000)
    full_dataset = full_dataset.cache()
    print("Size of Training Dataset: ", len(list(full_dataset)))
    
    feature_dictionary = {
    'label': tensorflow.io.FixedLenFeature([], tensorflow.int64),
    'label_normal': tensorflow.io.FixedLenFeature([], tensorflow.int64),
    'image': tensorflow.io.FixedLenFeature([], tensorflow.string)
    }   
 
    full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tensorflow.data.experimental.AUTOTUNE)
    print(full_dataset)
    for image_features in full_dataset:
        image = image_features['image'].numpy()
        image = tensorflow.io.decode_raw(image_features['image'], tensorflow.uint8)
        image = tensorflow.reshape(image, [299, 299])        
        image=image.numpy()
        image=cv2.resize(image,(Image_height,Image_width))
        image=cv2.merge([image,image,image])        
        
        # Generate file path
        file_name = 'image_{}.png'.format(np.random.randint(1, 1000000))  # Unique filename
        temp_image_path = os.path.join('/tmp/', file_name)  # Save image temporarily
        cv2.imwrite(temp_image_path, image)  # Save the image to disk

        # Determine the path based on label
        label = image_features['label'].numpy()
        if label == 0:
            destination_path = os.path.join(negative_path, file_name)
        elif label == 1:
            destination_path = os.path.join(benign_classification_path, file_name)
        elif label == 2:
            destination_path = os.path.join(benign_mass_path, file_name)
        elif label == 3:
            destination_path = os.path.join(malignant_classification_path, file_name)
        else:
            destination_path = os.path.join(malignant_mass_path, file_name)
        
        # Copy the file to the destination directory
        shutil.copyfile(temp_image_path, destination_path)
        
        # Optionally, you can remove the temporary file
        os.remove(temp_image_path)

filenames = [
    '/kaggle/input/ddsm-mammography/training10_0/training10_0.tfrecords',
    '/kaggle/input/ddsm-mammography/training10_1/training10_1.tfrecords',
    '/kaggle/input/ddsm-mammography/training10_2/training10_2.tfrecords',
    '/kaggle/input/ddsm-mammography/training10_3/training10_3.tfrecords',
    '/kaggle/input/ddsm-mammography/training10_4/training10_4.tfrecords'
]
 
for file in filenames:
    read_data(file)

split_copy(negative_path, os.path.join(train_dir, 'negative'), os.path.join(val_dir, 'negative'))
split_copy(benign_classification_path, os.path.join(train_dir, 'benign_classification'), os.path.join(val_dir, 'benign_classification'))
split_copy(benign_mass_path, os.path.join(train_dir, 'benign_mass'), os.path.join(val_dir, 'benign_mass'))
split_copy(malignant_classification_path, os.path.join(train_dir, 'malignant_classification'), os.path.join(val_dir, 'malignant_classification'))
split_copy(malignant_mass_path, os.path.join(train_dir, 'malignant_mass'), os.path.join(val_dir, 'malignant_mass'))

# Breast histopathology datasets
# New destination directory
destination_dir = '/client2/histopathological'

# Function to copy specific folders to a new location
def copy_images(src_base_dir, dest_dir, subdirs, magnifications):
    os.makedirs(dest_dir, exist_ok=True)
    for subdir in subdirs:
        dest_folder = os.path.join(dest_dir, subdir)
        sub_subdir = os.listdir(os.path.join(src_base_dir, subdir))
        os.makedirs(dest_folder, exist_ok=True)
        for case in sub_subdir:
            for mag in magnifications:
                src_folder = os.path.join(src_base_dir, subdir, case, mag)
                if os.path.exists(src_folder):
                    for filename in os.listdir(src_folder):
                        src_file = os.path.join(src_folder, filename)
                        dest_file = os.path.join(dest_folder, filename)
                        shutil.copy2(src_file, dest_file)
                else:
                    print(f"Folder {src_folder} does not exist.")


# Step 3: Copy the relevant directories to the new location
source_dir = './BreaKHis_v1/histology_slides/breast'
# folders_to_copy = ['benign/SOB', 'malignant/SOB']

# Define the specific subdirectories and magnifications
subdirs_beg = ['adenosis', 'fibroadenoma', 'phyllodes_tumor', 'tubular_adenoma']
magnifications = ['40X', '100X', '200X', '400X']

subdirs_mal = ['ductal_carcinoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma']
magnifications = ['40X', '100X', '200X', '400X']

copy_images(source_dir+'/benign/SOB', destination_dir+'/benign', subdirs_beg, magnifications)
copy_images(source_dir+'/malignant/SOB', destination_dir+'/malignant', subdirs_mal, magnifications)

# benign category
adenosis_path = os.path.join(destination_dir+'/benign/adenosis')
fibroadenoma_classification_path = os.path.join(destination_dir+'/benign/fibroadenoma')
phyllodes_tumor_path = os.path.join(destination_dir+'/benign/phyllodes_tumor')
tubular_adenoma_path = os.path.join(destination_dir+'/benign/tubular_adenoma')

# malignant
ductal_carcinoma_path = os.path.join(destination_dir+'/malignant/ductal_carcinoma')
lobular_carcinoma_path = os.path.join(destination_dir+'/malignant/lobular_carcinoma')
mucinous_carcinoma_path = os.path.join(destination_dir+'/malignant/mucinous_carcinoma')
papillary_carcinoma_path = os.path.join(destination_dir+'/malignant/papillary_carcinoma')

# benign split 
split_copy(adenosis_path, os.path.join(train_dir, 'adenosis'), os.path.join(val_dir, 'adenosis'))
split_copy(fibroadenoma_classification_path, os.path.join(train_dir, 'fibroadenoma'), os.path.join(val_dir, 'fibroadenoma'))
split_copy(phyllodes_tumor_path, os.path.join(train_dir, 'phyllodes_tumor'), os.path.join(val_dir, 'phyllodes_tumor'))
split_copy(tubular_adenoma_path, os.path.join(train_dir, 'tubular_adenoma'), os.path.join(val_dir, 'tubular_adenoma'))

# malignant
split_copy(ductal_carcinoma_path, os.path.join(train_dir, 'ductal_carcinoma'), os.path.join(val_dir, 'ductal_carcinoma'))
split_copy(lobular_carcinoma_path, os.path.join(train_dir, 'lobular_carcinoma'), os.path.join(val_dir, 'lobular_carcinoma'))
split_copy(mucinous_carcinoma_path, os.path.join(train_dir, 'mucinous_carcinoma'), os.path.join(val_dir, 'mucinous_carcinoma'))
split_copy(papillary_carcinoma_path, os.path.join(train_dir, 'papillary_carcinoma'), os.path.join(val_dir, 'papillary_carcinoma'))

print(countingFilesInDirectory(train_dir))
print(countingFilesInDirectory(val_dir))

# Load CSV data
def load_data_csv():
    # Load the dataset
    dicom_data_csv = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv')
    image_directory = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/jpeg'
    
    dicom_data_csv.drop(['PatientBirthDate','AccessionNumber','Columns','ContentDate','ContentTime','PatientSex','PatientBirthDate',
        'ReferringPhysicianName','Rows','SOPClassUID','SOPInstanceUID',
        'StudyDate','StudyID','StudyInstanceUID',
        'StudyTime','InstanceNumber','SeriesInstanceUID','SeriesNumber'],axis =1, inplace=True) 
    
    # selecting images based on series description.
    cropped_images=dicom_data_csv[dicom_data_csv.SeriesDescription=='cropped images'].image_path
    full_mammo_images=dicom_data_csv[dicom_data_csv.SeriesDescription=='full mammogram images'].image_path
    ROI_mask_images=dicom_data_csv[dicom_data_csv.SeriesDescription=='ROI mask images'].image_path

    # updating image directory.
    cropped_images=cropped_images.replace('CBIS-DDSM/jpeg',image_directory, regex=True)
    full_mammo_images=full_mammo_images.replace('CBIS-DDSM/jpeg',image_directory,regex=True)
    ROI_mask_images=ROI_mask_images.replace('CBIS-DDSM/jpeg',image_directory,regex=True)
    
    # view new paths
    print('Cropped Images paths:\n')
    print(cropped_images.iloc[0])
    print('Full mammo Images paths:\n')
    print(full_mammo_images.iloc[0])
    print('ROI Mask Images paths:\n')
    print(ROI_mask_images.iloc[0])
    
    full_mammo_images_dict=dict()
    cropped_images_dict=dict()
    ROI_mask_images_dict=dict()

    for data1 in full_mammo_images:
        key=data1.split("/")[5]
        full_mammo_images_dict[key]=data1 
    for data1 in cropped_images:
        key=data1.split("/")[5]
        cropped_images_dict[key]=data1   
    for data1 in ROI_mask_images:
        key=data1.split("/")[5]
        ROI_mask_images_dict[key]=data1 
    # view keys
    next(iter((full_mammo_images_dict.items())))
    
    dicom_cleaned_data = dicom_data_csv.copy()
    dicom_cleaned_data.head()
    dicom_cleaned_data['SeriesDescription'].fillna(method = 'bfill', axis = 0, inplace=True)
    dicom_cleaned_data['Laterality'].fillna(method = 'bfill', axis = 0, inplace=True)
    
    # mass case train.
    print("mass_case_df")
    mass_case_df = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv')
    mass_case_df.head()
    mass_case_df = mass_case_df.rename(columns={'left or right breast':'left_or_right_breast',
    'image view':'image_view','abnormality id':'abnormality_id','mass shape':'mass_shape',
    'mass margins':'mass_margins','image file path':'image_file_path', 'breast density': 'breast_density',
    'cropped image file path':'cropped_image_file_path', 'calc type': 'calc_type', 
    'calc distribution':'calc_distribution', 'abnormality type':'abnormality_type',
    'ROI mask file path':'ROI_mask_file_path'})


    mass_case_df['left_or_right_breast'] = mass_case_df['left_or_right_breast'].astype('category')
    mass_case_df['image_view'] = mass_case_df['image_view'].astype('category')
    mass_case_df['abnormality_type'] = mass_case_df['abnormality_type'].astype('category')
    mass_case_df['pathology'] = mass_case_df['pathology'].astype('category')
    print("Is null check (before data cleaning): ",mass_case_df.isna().sum())

    mass_case_df.isna().sum()
    
    mass_case_df['mass_shape'] = mass_case_df['mass_shape'].bfill() 
    mass_case_df['mass_margins'] = mass_case_df['mass_margins'].bfill() 

    print("Is null check (after data cleaning): ",mass_case_df.isna().sum())
    
    # mass case test.
    print("mass_case_test_df")
    mass_case_test_df = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv')
    mass_case_test_df.head()
    
    mass_case_test_df = mass_case_test_df.rename(columns={'left or right breast':'left_or_right_breast',
    'image view':'image_view','abnormality id':'abnormality_id','mass shape':'mass_shape',
    'mass margins':'mass_margins','image file path':'image_file_path', 'breast density': 'breast_density',
    'cropped image file path':'cropped_image_file_path', 'calc type': 'calc_type', 
    'calc distribution':'calc_distribution', 'abnormality type':'abnormality_type',
    'ROI mask file path':'ROI_mask_file_path'})


    mass_case_test_df['left_or_right_breast'] = mass_case_test_df['left_or_right_breast'].astype('category')
    mass_case_test_df['image_view'] = mass_case_test_df['image_view'].astype('category')
    mass_case_test_df['abnormality_type'] = mass_case_test_df['abnormality_type'].astype('category')
    mass_case_test_df['pathology'] = mass_case_test_df['pathology'].astype('category')
    print("Is null check (before data cleaning): ",mass_case_test_df.isna().sum())

    
    mass_case_test_df['mass_margins'] = mass_case_test_df['mass_margins'].bfill()

    print("Is null check (after data cleaning): ",mass_case_test_df.isna().sum())

    # fixing image path
    def fix_image_path(data):
        for i, img in enumerate(data.values):
            img_name=img[11].split("/")[2]
            if img_name in full_mammo_images_dict:
                data.iloc[i,11]=full_mammo_images_dict[img_name]
                
            img_name=img[12].split("/")[2]
            if img_name in cropped_images_dict:
                data.iloc[i,12]=cropped_images_dict[img_name]
            
            img_name = img[13].split("/")[2]
            if img_name in ROI_mask_images_dict:
                data.iloc[i, 13] = ROI_mask_images_dict[img_name]
                  
    fix_image_path(mass_case_df)
    fix_image_path(mass_case_test_df)
    
    value = mass_case_df['pathology'].value_counts()
    print("Pathology Type: ")
    print(value)
    
    def image_processor(image_path, target_size):
        """Preprocess images for the model"""
        absolute_image_path = os.path.abspath(image_path)
        image = cv2.imread(absolute_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (target_size[1], target_size[0]))
        pil_image = Image.fromarray(image)
        
        # Apply enhancements
        pil_image = ImageEnhance.Color(pil_image).enhance(1.35)
        pil_image = ImageEnhance.Contrast(pil_image).enhance(1.45)
        pil_image = ImageEnhance.Sharpness(pil_image).enhance(2.5)
        
        # Define color jitter transformations
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        # Apply color jitter transformation
        transformed_image = color_jitter(pil_image)
        
        # Convert the PIL image back to a NumPy array
        image_array = np.array(transformed_image)
        
        # Normalize the image array
        image_array = image_array / 255.0
        
        # Data Augmentation using ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True
        )
        image_array = datagen.random_transform(image_array)
        
        return image_array
    
    full_mass_df = pd.concat([mass_case_df, mass_case_test_df], axis=0)
    print(full_mass_df.shape)
    
    # Define the target size
    target_size = (224, 224, 3)

    # Apply preprocessor to train data
    full_mass_df['processed_images'] = full_mass_df['image_file_path'].apply(lambda x: image_processor(x, target_size))
    
    mapper={'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}

    XX=np.array(full_mass_df['processed_images'].tolist())
    
    # List of columns to drop
    columns_to_drop = ['patient_id', 'abnormality_id','assessment','subtlety']

    # Drop the specified columns
    full_mass_df = full_mass_df.drop(columns=columns_to_drop)
    print(full_mass_df.head())
                
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
class FlowerClient2(fl.client.NumPyClient):
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
        print("Local Training Metrics on client 2: {}".format(results))
        results_list.append(results)
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.model.evaluate(
            [self.val_img_gen.next()[0], self.val_csv], self.val_csv_labels
        )
        num_examples_test = len(self.val_csv)
        
        print("Evaluation accuracy on Client 2 after weight aggregation : ", accuracy)
        return loss, num_examples_test, {"accuracy": accuracy}

# Load data
train_generator, val_generator = load_data()

# Create model
model = get_combined_model()

# # Start Flower client
# fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient2(model, x_train, y_train, x_test, y_test))

plotClientData(results_list)