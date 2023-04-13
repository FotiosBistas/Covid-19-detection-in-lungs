import os
import numpy as np 
import random
from shutil import copyfile

# Set the paths to your directories
train_dir = 'train'
val_dir = 'validation'
test_dir = 'test'

# Create the validation and test directories if they do not exist
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create lists of file paths for the covid and non-covid images
covid_files = [os.path.join(train_dir, 'covid', f) for f in os.listdir(os.path.join(train_dir, 'covid'))]
non_covid_files = [os.path.join(train_dir, 'non_covid', f) for f in os.listdir(os.path.join(train_dir, 'non_covid'))]


covid_files = np.array(covid_files)
non_covid_files = np.array(non_covid_files)


validation_percentage = 10
testing_percentage = 10


covid_validation_number = int(len(covid_files) * (validation_percentage/100))
non_covid_validation_number = int(len(non_covid_files) * (validation_percentage/100))


covid_testing_number = int(len(covid_files) * (testing_percentage/100))
non_covid_testing_number = int(len(non_covid_files) * (testing_percentage/100))


print("Covid validation number:" ,covid_validation_number)
print("Covid testing number:" ,covid_testing_number)
print("Non covid validation number:" ,non_covid_validation_number)
print("Non covid testing number:", non_covid_testing_number)



covid_validation_files = covid_files[:covid_validation_number]
covid_testing_files = covid_files[covid_validation_number: (covid_validation_number + covid_testing_number)]
covid_training_files = covid_files[(covid_validation_number + covid_testing_number): len(covid_files)]



print(len(covid_testing_files))
print(len(covid_validation_files))
print(len(covid_training_files))

print(len(non_covid_files))

non_covid_validation_files = non_covid_files[:non_covid_validation_number]
non_covid_testing_files = non_covid_files[non_covid_validation_number: (non_covid_validation_number + non_covid_testing_number)]
non_covid_training_files = non_covid_files[(non_covid_validation_number + non_covid_testing_number) : len(non_covid_files)]


print(len(non_covid_testing_files))
print(len(non_covid_validation_files))
print(len(non_covid_training_files))