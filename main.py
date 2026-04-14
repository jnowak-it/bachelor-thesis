import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

# --- DATA PREPARATION ---
path_normal = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/normal_dogs"
path_my_dog = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/my_dog"
path_similar= r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/similar_dogs"
path_different =r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/different_dogs"

def load_images(folder_path, label):
    """ Loading the images from provided folder and label them.
     Args:
         folder_path (string): Path to the folder containing subfolders.
         label (int): Class label ( 0 = my dog (Kiara), 1 = not my dog)
     Returns:
         images (list): List of normalized images (224x224x3).
         labels (list): List of labels according to the image.
         """
    images = []
    labels = []
    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                if os.path.isfile(file_path) and file_name.endswith(".jpg"): # avoiding the system files (.DS_Store) and other non-images
                    img = cv2.imread(file_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224)) # shape that is mandatory to use with ResNet-50 architecture
                        img = img / 255.0
                        images.append(img)
                        labels.append(label)
                    else:
                        print(f"There is a problem with: {file_name}")
    return images, labels


kiara_images, kiara_labels = load_images(path_my_dog, 0)
similar_images, similar_labels = load_images(path_similar, 1)
normal_images, normal_labels = load_images(path_normal, 1)
different_images, different_labels = load_images(path_different, 1)

kiara_detailed = [0] * len(kiara_images)
similar_detailed = [1] * len(similar_images)
normal_detailed = [2] * len(normal_images)
different_detailed = [3] * len(different_images)

X = kiara_images + similar_images + normal_images + different_images
y = kiara_labels + similar_labels + normal_labels + different_labels
y_detailed = kiara_detailed + similar_detailed + normal_detailed + different_detailed

X = np.array(X)
y = np.array(y)
y_detailed = np.array(y_detailed)

print(X.shape)
print(y.shape)
print(y_detailed.shape)

y_det_train, y_det_test = train_test_split(y_detailed, test_size=0.15, random_state=42, stratify=y)
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.18, random_state=42, stratify=y_trainval)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print(X_val.shape)
print(y_val.shape)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_val, return_counts=True))
print(np.unique(y_test, return_counts=True))
print(np.unique(y_detailed, return_counts=True))

# --- MODEL ARCHITECTURE ---

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
print(base_model.count_params())

inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])














