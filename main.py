import os
from random import shuffle

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam

# --- DATA PREPARATION ---
path_normal = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/normal_dogs"
path_my_dog = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/my_dog"
path_similar= r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/similar_dogs"
path_different =r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/different_dogs"

my_dog_standard = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/my_dog/my_dog_standard"
my_dog_young = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/my_dog/my_dog_young"
my_dog_difficult = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset/my_dog/my_dog_difficult"

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


# kiara_images, kiara_labels = load_images(path_my_dog, 0)
similar_images, similar_labels = load_images(path_similar, 1)
normal_images, normal_labels = load_images(path_normal, 1)
different_images, different_labels = load_images(path_different, 1)

def load_images_flat(subfolder_path, label):
    images_flat = []
    labels_flat = []
    for file_name in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".jpg"): # avoiding the system files (.DS_Store) and other non-images
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224)) # shape that is mandatory to use with ResNet-50 architecture
                img = img / 255.0
                images_flat.append(img)
                labels_flat.append(label)
            else:
                print(f"There is a problem with: {file_name}")
    return images_flat, labels_flat

standard_images, standard_labels = load_images_flat(my_dog_standard, 0)
difficult_images, difficult_labels = load_images_flat(my_dog_difficult, 0)
young_images, young_labels = load_images_flat(my_dog_young, 0)

# kiara_detailed = [0] * len(kiara_images)
standard_detailed = [10] * len(standard_images)
young_detailed = [11] * len(young_images)
difficult_detailed = [12] * len(difficult_images)
similar_detailed = [1] * len(similar_images)
normal_detailed = [2] * len(normal_images)
different_detailed = [3] * len(different_images)

X = standard_images + young_images + difficult_images + similar_images + normal_images + different_images
y = standard_labels + young_labels + difficult_labels + similar_labels + normal_labels + different_labels
y_detailed = standard_detailed + young_detailed + difficult_detailed + similar_detailed + normal_detailed + different_detailed

X = np.array(X)
y = np.array(y)
y_detailed = np.array(y_detailed)

print(X.shape)
print(y.shape)
print(y_detailed.shape)

# Pierwszy split: trainval / test
indices = np.arange(len(X))
idx_train_val, idx_test = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)

X_train_val, X_test = X[idx_train_val], X[idx_test]
y_train_val, y_test = y[idx_train_val], y[idx_test]
y_det_trainval = y_detailed[idx_train_val]
y_det_test = y_detailed[idx_test]

# Drugi split: train / val
indices_tv = np.arange(len(X_train_val))
idx_train, idx_val = train_test_split(indices_tv, test_size=0.18, random_state=42, stratify=y_train_val)

X_train, X_val = X_train_val[idx_train], X_train_val[idx_val]
y_train, y_val = y_train_val[idx_train], y_train_val[idx_val]
y_det_train = y_det_trainval[idx_train]

# indices = np.arange(len(X))
# idx_train_val, idx_test = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)
# X_train_val, X_test = X[idx_train_val], X[idx_test]
# y_train_val, y_test = y[idx_train_val], y[idx_test]
# y_det_test = y_detailed[idx_test]
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=42, stratify=y_train_val)
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

data_augmentation = tf.keras.Sequential([RandomFlip("horizontal"), RandomRotation(0.1), RandomZoom(0.1), RandomBrightness(0.1)])
new_photos = []
new_labels = []
for i in range(len(X_train)):
    if y_det_train[i] == 10:
            for p in range(4):
                photo = X_train[i]
                photo_dimension = tf.expand_dims(photo, axis=0)
                photo_augmented = data_augmentation(photo_dimension)
                new_photos.append(photo_augmented[0])
                new_labels.append(0)
    elif y_det_train[i] == 11:
            for p in range(2):
                photo = X_train[i]
                photo_dimension = tf.expand_dims(photo, axis=0)
                photo_augmented = data_augmentation(photo_dimension)
                new_photos.append(photo_augmented[0])
                new_labels.append(0)
    elif y_det_train[i] == 12:
            for p in range(1):
                photo = X_train[i]
                photo_dimension = tf.expand_dims(photo, axis=0)
                photo_augmented = data_augmentation(photo_dimension)
                new_photos.append(photo_augmented[0])
                new_labels.append(0)
X_train = np.concatenate([X_train, np.array(new_photos)])
y_train = np.concatenate([y_train, np.array(new_labels)])
shuffled_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffled_indices]
y_train = y_train[shuffled_indices]
print(np.unique(y_train, return_counts=True))

weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(zip(np.unique(y_train), weight))
print(class_weight)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = inputs
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping], validation_data=(X_val, y_val), class_weight=class_weight)

model.save('model_01.keras')

base_model.trainable = True
for i, layer in enumerate(base_model.layers):
    if i < len(base_model.layers) - 10:
        layer.trainable = False

# adam_test = Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)
# model.compile(optimizer=adam_test, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
# model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping], validation_data=(X_val, y_val), class_weight=class_weight)
# model.save('model_02.keras')

# early_stopping_ft = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto', restore_best_weights=True)
# adam_test = Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.999)
# model.compile(optimizer=adam_test, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
# model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping_ft], validation_data=(X_val, y_val), class_weight=class_weight)
# model.save('model_03.keras')

# adam_test = Adam(learning_rate=5e-6, beta_1=0.9, beta_2=0.999)
# model.compile(optimizer=adam_test, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
# model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping_ft], validation_data=(X_val, y_val), class_weight=class_weight)
# model.save('model_04.keras')

model_evaluate = model.evaluate(X_test, y_test)
print(model_evaluate)

model_predict = model.predict(X_test)
model_predict = model_predict.flatten()
model_predict = model_predict > 0.5
model_predict = model_predict.astype(int)

conf_matrix = confusion_matrix(y_test, model_predict)
print(conf_matrix)
class_report = classification_report(y_test, model_predict, target_names=['Kiara', 'Nie Kiara'])
print(class_report)

similar = np.mean(model_predict[y_det_test == 1])
print(f'Similar dogs {similar}')
normal = np.mean(model_predict[y_det_test == 2])
print(f'Normal dogs {normal}')
different = np.mean(model_predict[y_det_test == 3])
print(f'Different dogs: {different}')






