import os
from os import listdir
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2
import tensorflow as tf
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from PIL import Image
from PIL import ImageEnhance
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Input
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.optimizers import Adam


mass_case_train = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv'
mass_case_test = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv'

print(mass_case_train.head())
print(mass_case_test.head())

mass_case_train = mass_case_train.rename(columns={'left or right breast':'left_or_right_breast',
'image view':'image_view','abnormality id':'abnormality_id','mass shape':'mass_shape',
'mass margins':'mass_margins','image file path':'image_file_path', 'breast density': 'breast_density',
'cropped image file path':'cropped_image_file_path', 'calc type': 'calc_type', 
 'calc distribution':'calc_distribution', 'abnormality type':'abnormality_type',
'ROI mask file path':'ROI_mask_file_path'})

mass_case_train['left_or_right_breast'] = mass_case_train['left_or_right_breast'].astype('category')
mass_case_train['image_view'] = mass_case_train['image_view'].astype('category')
mass_case_train['abnormality_type'] = mass_case_train['abnormality_type'].astype('category')
mass_case_train['pathology'] = mass_case_train['pathology'].astype('category')

print(mass_case_train.isna().sum())

mass_case_train['mass_shape'] = mass_case_train['mass_shape'].bfill() 
mass_case_train['mass_margins'] = mass_case_train['mass_margins'].bfill() 

print(mass_case_train.isna().sum())

mass_case_test = mass_case_test.rename(columns={'left or right breast':'left_or_right_breast',
'image view':'image_view','abnormality id':'abnormality_id','mass shape':'mass_shape',
'mass margins':'mass_margins','image file path':'image_file_path', 'breast density': 'breast_density',
'cropped image file path':'cropped_image_file_path', 'calc type': 'calc_type', 
 'calc distribution':'calc_distribution', 'abnormality type':'abnormality_type',
'ROI mask file path':'ROI_mask_file_path'})

mass_case_test['left_or_right_breast'] = mass_case_test['left_or_right_breast'].astype('category')
mass_case_test['image_view'] = mass_case_test['image_view'].astype('category')
mass_case_test['abnormality_type'] = mass_case_test['abnormality_type'].astype('category')
mass_case_test['pathology'] = mass_case_test['pathology'].astype('category')

print(mass_case_test.isna().sum())

mass_case_test['mass_margins'] = mass_case_test['mass_margins'].bfill()

print(mass_case_test.isna().sum())

print(f'Shape of mass_train: {mass_case_train.shape}')
print(f'Shape of mass_test: {mass_case_test.shape}')

dicom_data_csv = pd.read_csv('/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv')
image_directory = '/kaggle/input/cbis-ddsm-breast-cancer-image-dataset/jpeg'
print(dicom_data_csv.head())

dicom_data_csv.drop(['PatientBirthDate','AccessionNumber','Columns','ContentDate','ContentTime','PatientSex','PatientBirthDate',
                                                'ReferringPhysicianName','Rows','SOPClassUID','SOPInstanceUID',
                                                'StudyDate','StudyID','StudyInstanceUID',
                     'StudyTime','InstanceNumber','SeriesInstanceUID','SeriesNumber'],axis =1, inplace=True) 

print(dicom_data_csv.info())

cropped_images=dicom_data_csv[dicom_data_csv.SeriesDescription=='cropped images'].image_path
full_mammo_images=dicom_data_csv[dicom_data_csv.SeriesDescription=='full mammogram images'].image_path
ROI_mask_images=dicom_data_csv[dicom_data_csv.SeriesDescription=='ROI mask images'].image_path

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
            
fix_image_path(mass_case_train)
fix_image_path(mass_case_test)

value = mass_case_train['pathology'].value_counts()
print(value)

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# create function to display images
def display_images(column, number):
    # create figure and axes
    number_to_visualize = number
    rows = 1
    cols = number_to_visualize
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
     
    # Loop through rows and display images
    for index, row in mass_case_train.head(number_to_visualize).iterrows():
        image_path = row[column]
        image = mpimg.imread(image_path)
        ax = axes[index]
        ax.imshow(image, cmap='gray')
        ax.set_title(f"{row['pathology']}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    
full_mass_df = pd.concat([mass_case_train, mass_case_test], axis=0)
print(full_mass_df.shape)

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


# Define the target size
target_size = (224, 224, 3)

# Apply preprocessor to train data
full_mass_df['processed_images'] = full_mass_df['image_file_path'].apply(lambda x: image_processor(x, target_size))

mapper={'MALIGNANT': 1, 'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0}
full_mass_df['pathology'] = full_mass_df['pathology'].replace(mapper)

# List of columns to drop
columns_to_drop = ['patient_id', 'abnormality_id','assessment','subtlety', 'image_file_path', 'cropped_image_file_path', 'ROI_mask_file_path']

# Drop the specified columns
full_mass_df = full_mass_df.drop(columns=columns_to_drop)

categorical_columns = ['left_or_right_breast', 'image_view', 'abnormality_type', 'mass_shape', 'mass_margins']
label_encoders = {}
    
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    full_mass_df[col] = label_encoders[col].fit_transform(full_mass_df[col])

print(full_mass_df.head())

vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def extract_features(processed_image):
    image = preprocess_input(processed_image)
    features = vgg16.predict(image)
    return features.flatten()

preprocessed_images = full_mass_df['processed_images'].to_numpy()
image_features_list = [extract_features(preprocessed_image) for preprocessed_image in preprocessed_images]
image_features = np.array(image_features_list)

csv_data  = full_mass_df.copy()
csv_data = csv_data.drop(columns=['processed_images', 'pathology'])

scaler = StandardScaler()
csv_data = pd.DataFrame(scaler.fit_transform(csv_data), columns=csv_data.columns)
csv_features = csv_data.values

# Combine CSV and image features
combined_features = np.hstack((csv_features, image_features))

input_shape = combined_features.shape[1]

input_layer = Input(shape=(input_shape,))
dense_layer_1 = Dense(512, activation='relu')(input_layer)
dropout_layer = Dropout(0.5)(dense_layer_1)
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

combined_model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
combined_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

labels = full_mass_df['pathology']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)


# Train the model
combined_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = combined_model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')