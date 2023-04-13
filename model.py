import tensorflow as tf 
from  tensorflow.keras import datasets,layers, models 
from tensorflow.keras.callbacks import ModelCheckpoint, History

import numpy as np 
import matplotlib.pyplot as plt 
from preprocess import train_dataset,test_dataset,validation_dataset


# Define the convnet architecture
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu',padding='same', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    layers.Conv2D(128, (3, 3), activation='relu',padding='same'),
    layers.BatchNormalization(), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Define callbacks
checkpoint = ModelCheckpoint('model_epoch_{epoch:02d}_val_loss_{val_loss:.2f}.h5', save_best_only=True,verbose=1)
history = History()

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

# Train the model
history = model.fit(train_dataset, epochs=20, validation_data=validation_dataset,
                    callbacks=[checkpoint, history])
# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_dataset)
print('Test accuracy:', test_acc)


