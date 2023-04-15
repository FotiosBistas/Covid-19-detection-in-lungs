import tensorflow as tf 
from  tensorflow.keras import datasets,layers, models 
from tensorflow.keras.callbacks import ModelCheckpoint, History, EarlyStopping, ReduceLROnPlateau

import numpy as np 
import matplotlib.pyplot as plt 
from SARS_CoV_2_CT.preprocess import train_dataset,test_dataset,validation_dataset


# Define the convnet architecture
model = tf.keras.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(192, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Define callbacks

#earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, min_delta=1e-4, mode='min')

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
    )

# Train the model
history = model.fit(train_dataset, epochs=10, validation_data=validation_dataset,
                    callbacks=[mcp_save,reduce_lr_loss])

# Load the best model
best_model = tf.keras.models.load_model('.mdl_wts.hdf5')

# Evaluate the model on test data
test_loss, test_acc = best_model.evaluate(test_dataset)
print('Test accuracy:', test_acc)


