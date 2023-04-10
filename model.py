import tensorflow as tf 
from read_images import readImageNames 
import glob
from  tensorflow.keras import datasets,layers, models 

import numpy as np 
import matplotlib.pyplot as plt 



test_covid_image_names = readImageNames("CT_COVID_IMG_NAMES/testCT_COVID.txt")
train_covid_image_names = readImageNames("CT_COVID_IMG_NAMES/trainCT_COVID.txt")
validation_covid_images_names = readImageNames("CT_COVID_IMG_NAMES/valCT_COVID.txt")

test_non_covid_image_names = readImageNames("CT_NON_COVID_IMG_NAMES/testCT_NON_COVID.txt")
train_non_covid_image_names = readImageNames("CT_NON_COVID_IMG_NAMES/trainCT_NON_COVID.txt")
validation_non_covid_images_names = readImageNames("CT_NON_COVID_IMG_NAMES/valCT_NON_COVID.txt")


#create a dataset with classified images 1 that have covid and 0 for those that don't have covid
train_filenames_labels = [(filename, 1) for filename in train_covid_image_names] + [(filename, 0) for filename in train_non_covid_image_names]
test_filenames_labels = [(filename, 1) for filename in test_covid_image_names] + [(filename, 0) for filename in test_non_covid_image_names] 
validation_filenames_labels = [(filename, 1) for filename in validation_covid_images_names] + [(filename, 0) for filename in validation_non_covid_images_names] 

print(train_filenames_labels)
print(test_filenames_labels)
print(validation_filenames_labels)
#covid_image_paths = glob.glob("CT_COVID/CT_COVID");
#non_covid_image_paths = glob.glob("CT_NonCOVID");
#
#covid_image_dataset = tf.data.Dataset.from_tensor_slices(covid_image_paths)
#non_covid_image_dataset = tf.data.Dataset.from_tensor_slices(non_covid_image_paths)
#
#print(covid_image_dataset)
#print(non_covid_image_dataset)
#
#x = layers.Conv2D(filters = 32, kernel_size = 3, activation="relu")(x)
#x = layers.MaxPooling2D(pool_size = 2) (x)