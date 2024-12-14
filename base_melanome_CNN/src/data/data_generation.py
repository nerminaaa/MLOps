import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import yaml

with open('./params.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Data augmentation for training and testing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

import os

# Use absolute paths based on the project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

training_set = train_datagen.flow_from_directory(
    os.path.join(project_root, 'data/base_melanome_CNN'),
    target_size=tuple(config['imgsz']),
    batch_size=config['batch'],
    class_mode=config['labeling_mode']
)

test_set = test_datagen.flow_from_directory(
    os.path.join(project_root, 'test_set'),
    target_size=tuple(config['imgsz']),
    batch_size=config['batch'],
    class_mode=config['labeling_mode']
)

print("Dataset Information:")
print(f"Number of Training Samples: {training_set.samples}")
print(f"Number of Test Samples: {test_set.samples}")
print(f"Number of Classes: {training_set.num_classes}")
print("Class Labels:", list(training_set.class_indices.keys()))


# Get number of classes and class names
num_classes = training_set.num_classes
class_names = list(training_set.class_indices.keys())
