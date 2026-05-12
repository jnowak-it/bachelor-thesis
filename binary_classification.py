import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom, RandomBrightness
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# --- ENVIRONMENT SETUP ---
# Set seed for reproducibility of random operations
np.random.seed(42)
tf.random.set_seed(42)

# --- DATA PREPARATION & LOADING ---
print("--- LOADING DATA ---")

BASE_DIR = r"/Users/jnowak/PycharmProjects/bachelor-thesis/dataset"

# Paths to the external datasets - Stanford Dogs Dataset (Label 1)
path_normal = os.path.join(BASE_DIR, "normal_dogs")
path_similar = os.path.join(BASE_DIR, "similar_dogs")
path_different = os.path.join(BASE_DIR, "different_dogs")

# Paths to the target dog datasets - my own database (Label 0)
path_my_dog = os.path.join(BASE_DIR, "my_dog")
my_dog_standard = os.path.join(path_my_dog, "my_dog_standard")
my_dog_young = os.path.join(path_my_dog, "my_dog_young")
my_dog_difficult = os.path.join(path_my_dog, "my_dog_difficult")

def load_images(folder_path, label):
    """Loads images from subfolders (Stanford Dogs Dataset structure)."""
    images = []
    labels = []
    for folder_name in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, file_name)
                # Avoid system files and ensure correct extension
                if os.path.isfile(file_path) and file_name.endswith(".jpg"):
                    img = cv2.imread(file_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (224, 224)) # Resize to match ResNet50 input requirements
                        images.append(img)
                        labels.append(label)
                    else:
                        print(f"Warning: Issue with reading {file_name}")
    return images, labels

def load_images_flat(subfolder_path, label):
    """Loads images directly from a single folder (custom dog photos)."""
    images_flat = []
    labels_flat = []
    for file_name in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".jpg"):
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                images_flat.append(img)
                labels_flat.append(label)
            else:
                print(f"Warning: Issue with reading {file_name}")
    return images_flat, labels_flat

# Load external dog images (Label 1)
similar_images, similar_labels = load_images(path_similar, 1)
normal_images, normal_labels = load_images(path_normal, 1)
different_images, different_labels = load_images(path_different, 1)

# Load target dog images (Label 0)
standard_images, standard_labels = load_images_flat(my_dog_standard, 0)
difficult_images, difficult_labels = load_images_flat(my_dog_difficult, 0)
young_images, young_labels = load_images_flat(my_dog_young, 0)

# Create detailed labels for deeper evaluation
standard_detailed = [10] * len(standard_images)
young_detailed = [11] * len(young_images)
difficult_detailed = [12] * len(difficult_images)
similar_detailed = [1] * len(similar_images)
normal_detailed = [2] * len(normal_images)
different_detailed = [3] * len(different_images)

# Combine datasets
X = standard_images + young_images + difficult_images + similar_images + normal_images + different_images
y = standard_labels + young_labels + difficult_labels + similar_labels + normal_labels + different_labels
y_detailed = standard_detailed + young_detailed + difficult_detailed + similar_detailed + normal_detailed + different_detailed

X = np.array(X)
y = np.array(y)
y_detailed = np.array(y_detailed)

print(f"Total dataset shape: X={X.shape}, y={y.shape}")


# --- TRAIN / VAL / TEST SPLIT ---
print("\n--- DATA SPLITTING ---")

# Step 1: Split into Train+Val (85%) and Test (15%)
indices = np.arange(len(X))
idx_train_val, idx_test = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)

X_train_val, X_test = X[idx_train_val], X[idx_test]
y_train_val, y_test = y[idx_train_val], y[idx_test]
y_det_trainval = y_detailed[idx_train_val]
y_det_test = y_detailed[idx_test]

# Step 2: Split Train+Val into Train (approx 70% total) and Val (approx 15% total)
indices_tv = np.arange(len(X_train_val))
idx_train, idx_val = train_test_split(indices_tv, test_size=0.18, random_state=42, stratify=y_train_val)

X_train, X_val = X_train_val[idx_train], X_train_val[idx_val]
y_train, y_val = y_train_val[idx_train], y_train_val[idx_val]
y_det_train = y_det_trainval[idx_train]

print(f"Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")


# --- DATA AUGMENTATION ---
print("\n--- AUGMENTING MINORITY CLASS ---")

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    RandomRotation(0.1),
    RandomZoom(0.1),
    RandomBrightness(0.1)
])

new_photos, new_labels, new_det_labels = [], [], []

# Augment specific target classes to balance the dataset
for i in range(len(X_train)):
    if y_det_train[i] == 10: # Kiara Standard
        for _ in range(4):
            photo_aug = data_augmentation(tf.expand_dims(X_train[i], axis=0), training=True)
            new_photos.append(photo_aug[0])
            new_labels.append(0)
            new_det_labels.append(10)
    elif y_det_train[i] == 11: # Kiara Young
        for _ in range(2):
            photo_aug = data_augmentation(tf.expand_dims(X_train[i], axis=0), training=True)
            new_photos.append(photo_aug[0])
            new_labels.append(0)
            new_det_labels.append(11)
    elif y_det_train[i] == 12: # Kiara Difficult
        for _ in range(1):
            photo_aug = data_augmentation(tf.expand_dims(X_train[i], axis=0), training=True)
            new_photos.append(photo_aug[0])
            new_labels.append(0)
            new_det_labels.append(12)

# Merge augmented data with the original training data
X_train = np.concatenate([X_train, np.array(new_photos)])
y_train = np.concatenate([y_train, np.array(new_labels)])
y_det_train = np.concatenate([y_det_train, np.array(new_det_labels)])

# Shuffle the training dataset
shuffled_indices = np.random.permutation(len(X_train))
X_train = X_train[shuffled_indices]
y_train = y_train[shuffled_indices]
y_det_train = y_det_train[shuffled_indices]

# Compute class weights for handling data imbalance during training
weight = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(zip(np.unique(y_train), weight))


# --- MODEL ARCHITECTURE & FEATURE EXTRACTION ---
print("\n--- BUILDING & TRAINING BASE MODEL ---")

# Load ResNet50 as base model and freeze its weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build custom classification head
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.resnet50.preprocess_input(inputs) # ResNet50 specific preprocessing
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# Train only the top classification layers
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping], validation_data=(X_val, y_val), class_weight=class_weight)
model.save('model_base.keras')


# --- FINE-TUNING ---
print("\n--- FINE-TUNING THE MODEL ---")

# Unfreeze the base model and freeze all but the last 10 layers
base_model.trainable = True
for i, layer in enumerate(base_model.layers):
    if i < len(base_model.layers) - 10:
        layer.trainable = False

# Recompile with a lower learning rate to avoid catastrophic forgetting
adam_ft = Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.999)
model.compile(optimizer=adam_ft, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early_stopping_ft = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=30, callbacks=[early_stopping_ft], validation_data=(X_val, y_val), class_weight=class_weight)
model.save('model_finetuned.keras')


# --- EVALUATION ---
print("\n--- MODEL EVALUATION ---")

# Evaluate on the unseen test set
model_evaluate = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {model_evaluate[0]:.4f}, Test Accuracy: {model_evaluate[1]:.4f}, Test AUC: {model_evaluate[2]:.4f}")

# Generate predictions
model_predict_prob = model.predict(X_test, verbose=0).flatten()
# Threshold set empirically to 0.65
model_predict_binary = (model_predict_prob > 0.65).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, model_predict_binary))

print("\nClassification Report:")
print(classification_report(y_test, model_predict_binary, target_names=['Target Dog (0)', 'Other Dogs (1)']))

# Detailed evaluation across different subgroups
print("\nDetailed breakdown of False Positive rates by negative categories:")
similar = np.mean(model_predict_binary[y_det_test == 1] == 1)
print(f'Similar dogs correctly classified as "Other": {similar * 100:.2f}%')

normal = np.mean(model_predict_binary[y_det_test == 2] == 1)
print(f'Normal dogs correctly classified as "Other": {normal * 100:.2f}%')

different = np.mean(model_predict_binary[y_det_test == 3] == 1)
print(f'Different dogs correctly classified as "Other": {different * 100:.2f}%')