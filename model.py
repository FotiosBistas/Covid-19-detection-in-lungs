import tensorflow as tf 
from  tensorflow.keras import datasets,layers, models 

import numpy as np 
import matplotlib.pyplot as plt 




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