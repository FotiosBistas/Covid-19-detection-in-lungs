import tensorflow as tf 
from  tensorflow.keras import datasets,layers, models 

import numpy as np 
import matplotlib.pyplot as plt 
from preprocess import train_dataset,test_dataset,validation_dataset


# Define the convnet architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)



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