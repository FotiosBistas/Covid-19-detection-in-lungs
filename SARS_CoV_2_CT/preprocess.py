import tensorflow as tf 
import os 

from keras import layers 

batchSize = 64
imageSize = (224,224)


parent_dir = 'C:\\Users\\fotis\\GitHub\\Covid-19-detection-in-lungs\\SARS_CoV_2_CT\\'
test_directory = parent_dir + '/test'
train_directory = parent_dir + '/train'
validation_directory = parent_dir + '/validation'

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_directory,
    label_mode='binary',
    batch_size=batchSize,
    image_size=imageSize,
)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_directory,
    label_mode='binary',
    batch_size=batchSize,
    image_size= imageSize,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_directory,
    label_mode='binary',
    batch_size=batchSize,
    image_size=imageSize,
)

rescale = tf.keras.layers.Rescaling(
    scale = 1./255,
)

subset_size = 0.3
augment_data = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
    ]
)


augmented_data = train_dataset.map(lambda image,label: (augment_data(image), label))
augmented_data = augmented_data.map(lambda image,label: (rescale(image), label))
augmented_data = augmented_data.take(3)
train_dataset = train_dataset.map(lambda image,label: (rescale(image), label))
train_dataset = train_dataset.concatenate(augmented_data)
train_dataset = train_dataset.shuffle(400)

validation_dataset = validation_dataset.map(lambda image,label: (rescale(image), label))
test_dataset = test_dataset.map(lambda image,label: (rescale(image), label))

