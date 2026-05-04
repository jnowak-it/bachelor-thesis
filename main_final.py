import os
import cv2
import tensorflow as tf
import numpy as np

print("--- LOADING DATA ---")

BASE_DIR = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset"
different_dogs = os.path.join(BASE_DIR, "different_dogs")
normal_dogs = os.path.join(BASE_DIR, "normal_dogs")
similar_dogs = os.path.join(BASE_DIR, "similar_dogs")
my_dog = os.path.join(BASE_DIR, "my_dog")
my_dog_young = os.path.join(my_dog, "my_dog_young")
my_dog_standard = os.path.join(my_dog, "my_dog_standard")
my_dog_difficult = os.path.join(my_dog, "my_dog_difficult")

def load_stanford_dataset(path, label):
    photos = []
    labels = []
    for breed in os.listdir(path):
        if os.path.isdir(os.path.join(path, breed)):
            for photo in os.listdir(os.path.join(path, breed)):
                if photo.endswith(".jpg"):
                    img = cv2.imread(os.path.join(path, breed, photo))
                    if img is not None:
                        img = cv2.resize(img, (224, 224))
                        img = img / 255.0
                        photos.append(img)
                        labels.append(label)
                    else:
                        print(f"There is no image {photo}")
                else:
                    print(f"Skipping {os.path.join(path, breed, photo)}")
        else:
            print(f"Warning: {os.path.join(path,breed)} is not a directory")
    return photos, labels

def load_my_dataset(path, label):
    photos = []
    labels = []
    for photo in os.listdir(path):
        if photo.endswith(".jpg"):
            img = cv2.imread(os.path.join(path, photo))
            if img is not None:
                img = cv2.resize(img, (224, 224))
                img = img / 255.0
                photos.append(img)
                labels.append(label)
            else:
                print(f"There is no image {photo}")
    return photos, labels

stanford_different, different_labels = load_stanford_dataset(different_dogs, 1)
stanford_normal, normal_labels = load_stanford_dataset(normal_dogs, 1)
stanford_similar, similar_labels = load_stanford_dataset(similar_dogs, 1)

kiara_young, young_labels = load_my_dataset(my_dog_young, 0)
kiara_standard, standard_labels = load_my_dataset(my_dog_standard, 0)
kiara_difficult, difficult_labels = load_my_dataset(my_dog_difficult, 0)

similar_detailed = [1] * len(similar_dogs)
normal_detailed = [2] * len(normal_dogs)
different_detailed = [3] * len(different_dogs)
standard_detailed = [10] * len(similar_dogs)
young_detailed = [11] * len(similar_dogs)
difficult_detailed = [12] * len(similar_dogs)

stanford_detailed = similar_detailed + normal_detailed + different_detailed
my_dog_detailed = standard_detailed + young_detailed + difficult_detailed

X = stanford_different + stanford_normal + stanford_similar + kiara_young + kiara_standard + kiara_difficult
y = different_labels + normal_labels + similar_labels + young_labels + difficult_labels + similar_labels
y_detailed = similar_detailed + normal_detailed + different_detailed + standard_detailed + young_detailed + difficult_detailed

X = np.array(X)
y = np.array(y)
y_detailed = np.array(y_detailed)

print(X.shape, y.shape, y_detailed.shape)