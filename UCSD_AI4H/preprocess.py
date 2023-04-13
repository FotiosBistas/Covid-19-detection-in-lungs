from read_images import readImageNames 
import tensorflow as tf 
from tensorflow.keras import datasets,layers 
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from read_images import loadImage
import numpy as np 
BATCH_SIZE =64



covid_directory = "C:\\Users\\fotis\\GitHub\\Covid-19-detection-in-lungs\\UCSD_AI4H\\CT_COVID\\"
non_covid_directory = "C:\\Users\\fotis\\GitHub\\Covid-19-detection-in-lungs\\UCSD_AI4H\\CT_NonCOVID\\"

test_covid_image_names = readImageNames("CT_COVID_IMG_NAMES/testCT_COVID.txt")
train_covid_image_names = readImageNames("CT_COVID_IMG_NAMES/trainCT_COVID.txt")
validation_covid_images_names = readImageNames("CT_COVID_IMG_NAMES/valCT_COVID.txt")

test_non_covid_image_names = readImageNames("CT_NON_COVID_IMG_NAMES/testCT_NON_COVID.txt")
train_non_covid_image_names = readImageNames("CT_NON_COVID_IMG_NAMES/trainCT_NON_COVID.txt")
validation_non_covid_images_names = readImageNames("CT_NON_COVID_IMG_NAMES/valCT_NON_COVID.txt")


train_filenames = [(covid_directory + filename) for filename in train_covid_image_names] + [(non_covid_directory + filename) for filename in train_non_covid_image_names]
train_labels = [1 for _ in range(len(train_covid_image_names))] + [0 for _ in range(len(train_non_covid_image_names))]

test_filenames = [(covid_directory + filename) for filename in test_covid_image_names] + [(non_covid_directory + filename) for filename in test_non_covid_image_names]
test_labels = [1 for _ in range(len(test_covid_image_names))] + [0 for _ in range(len(test_non_covid_image_names))]

validation_filenames = [(covid_directory + filename) for filename in validation_covid_images_names] + [(non_covid_directory + filename) for filename in validation_non_covid_images_names]
validation_labels = [1 for _ in range(len(validation_covid_images_names))] + [0 for _ in range(len(validation_non_covid_images_names))]


train_filenames = np.array(train_filenames)
train_labels = np.array(train_labels)

test_filenames = np.array(test_filenames)
test_labels = np.array(test_labels)

validation_filenames = np.array(validation_filenames)
validation_labels = np.array(validation_labels)

train_images = np.array(list(map(loadImage,train_filenames)))
test_images = np.array(list(map(loadImage, test_filenames)))
validation_images = np.array(list(map(loadImage, validation_filenames)))


train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))


train_dataset = train_dataset.shuffle(buffer_size=len(train_images))
test_dataset = test_dataset.shuffle(buffer_size=len(test_images))
validation_dataset = validation_dataset.shuffle(buffer_size=len(validation_dataset))

train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)
validation_dataset = validation_dataset.batch(BATCH_SIZE)

#train_dataset = train_dataset.batch(BATCH_SIZE)
#train_dataset = train_dataset.map()


#data_augmentation = tf.keras.Sequential([
#  layers.RandomFlip("horizontal_and_vertical"),
#  layers.RandomRotation(0.2),
#])
#
#train_dataset_augmented = train_dataset.map(lambda x, y: (data_augmentation(x), y))
#
#train_dataset = train_dataset.concatenate(train_dataset_augmented)

for image,label in train_dataset.take(1): 
    print(image,label)