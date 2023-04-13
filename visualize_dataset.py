import matplotlib.pyplot as plt
from preprocess import train_dataset, test_dataset, validation_dataset, data_augmentation

def plot_images(dataset, datasetName ,num_images=10):
    plt.figure(figsize=(30, 30))
    subplot_index = 1
    for images, labels in dataset.take(1):
        for i in range(num_images):
            
            plt.subplot(5, 5, subplot_index)
            plt.imshow(images[i])
            plt.title("Label: {}".format(labels[i]))
            plt.axis("off")
            subplot_index += 1

    plt.suptitle('Sample images from ' + datasetName + " dataset")

# plot some images from each dataset
plot_images(train_dataset, "Training")
plot_images(test_dataset, "Testing")
plot_images(validation_dataset, "Validation")
plt.show()