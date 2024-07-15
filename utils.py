import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


def plotClientData(data):
    df = pd.DataFrame(data)
    plt.plot(df['accuracy'],color = 'b', label = 'training accuracy')
    plt.plot(df['val_accuracy'],color = 'g', label = 'validation accuracy')
    plt.legend(loc = 'lower right')
    plt.xlabel('Rounds')
    plt.ylabel('accuracy')
    plt.show()
    
def split_copy(class_dir, train_output_dir, val_output_dir):
    images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(val_output_dir, exist_ok=True)

    for img in train_images:
        shutil.copyfile(img, os.path.join(train_output_dir, os.path.basename(img)))

    for img in val_images:
        shutil.copyfile(img, os.path.join(val_output_dir, os.path.basename(img)))


def countingFilesInDirectory(directory):
    counts = {}
    for subdirectory in os.listdir(directory):
        subdirectory_path = os.path.join(directory, subdirectory)
        if os.path.exists(subdirectory_path) and os.path.isdir(subdirectory_path):
            # Count the number of files in the subdirectory
            file_count = len([f for f in os.listdir(subdirectory_path) if os.path.isfile(os.path.join(subdirectory_path, f))])
            counts[subdirectory] = file_count
    return counts