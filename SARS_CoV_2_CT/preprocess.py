import tensorflow as tf 
import os 



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

train_dataset = train_dataset.map(lambda image,label: (rescale(image), label))
validation_dataset = validation_dataset.map(lambda image,label: (rescale(image), label))
test_dataset = test_dataset.map(lambda image,label: (rescale(image), label))

