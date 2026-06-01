import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.applications import ResNet50
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
np.random.seed(42)

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
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

similar_detailed = [1] * len(stanford_similar)
normal_detailed = [2] * len(stanford_normal)
different_detailed = [3] * len(stanford_different)
standard_detailed = [10] * len(kiara_standard)
young_detailed = [11] * len(kiara_young)
difficult_detailed = [12] * len(kiara_difficult)

X = stanford_different + stanford_normal + stanford_similar + kiara_young + kiara_standard + kiara_difficult
y = different_labels + normal_labels + similar_labels + young_labels + standard_labels + difficult_labels
y_detailed = different_detailed + normal_detailed + similar_detailed + young_detailed + standard_detailed + difficult_detailed

X = np.array(X)
y = np.array(y)
y_detailed = np.array(y_detailed)

print(f"\nX size: {X.shape}\ny size: {y.shape}\ny_detailed size: {y_detailed.shape}")

# --- DATA VISUALIZATION ---

categories = [
    ("Stanford Similar", stanford_similar),
    ("Kiara Standard", kiara_standard),
]
plt.figure(figsize=(7, 4))
plt.suptitle("Przykładowe zdjęcia po przetworzeniu")
for i in range(len(categories)):
    name, photo = categories[i]
    for j in range(3):
        plt.subplot(2, 3, i * 3 + j + 1)
        plt.imshow(photo[j])
plt.show()

print("\n--- TRAIN / VAL / TEST DATA ---")

# First split: trainval / test
indices = np.arange(len(X))
train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)
X_train_val, X_test = X[train_val_idx], X[test_idx]
y_train_val, y_test = y[train_val_idx], y[test_idx]
y_train_val_detailed, y_test_detailed = y_detailed[train_val_idx], y_detailed[test_idx]

# Second split: train / val
indices_tv = np.arange(len(X_train_val))
train_idx, val_idx = train_test_split(indices_tv, test_size=0.18, random_state=42, stratify=y_train_val)
X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
y_train_detailed, y_val_detailed = y_train_val_detailed[train_idx], y_train_val_detailed[val_idx]

print(f"Train shape: {X_train.shape}, {y_train.shape}, {y_train_detailed.shape}")
print(f"Val shape: {X_val.shape}, {y_val.shape}, {y_val_detailed.shape}")
print(f"Test shape: {X_test.shape}, {y_test.shape}, {y_test_detailed.shape}")

print("\n PAIRS FOR SIAMESE NEURAL NETWORK")

def siamese_pairs(photos, labels):
    pairs_1 = []
    pairs_2 = []
    pair_labels = []
    my_dog_idx = np.where(labels == 0)[0]
    stanford_idx = np.where(labels == 1)[0]
    pairs = min(len(my_dog_idx), len(stanford_idx)) * 2
    for random_dog in range(pairs):
        rand_my_dog_idx = np.random.choice(my_dog_idx, 2, replace=False)
        pairs_1.append(photos[rand_my_dog_idx[0]])
        pairs_2.append(photos[rand_my_dog_idx[1]])
        pair_labels.append(1)
        rand_kiara_idx = np.random.choice(my_dog_idx, 1)[0]
        rand_stanford_idx = np.random.choice(stanford_idx, 1)[0]
        pairs_1.append(photos[rand_stanford_idx])
        pairs_2.append(photos[rand_kiara_idx])
        pair_labels.append(0)
    return np.array(pairs_1), np.array(pairs_2), np.array(pair_labels)

train_pairs_1, train_pairs_2, train_pair_labels = siamese_pairs(X_train, y_train)
val_pairs_1, val_pairs_2, val_pair_labels = siamese_pairs(X_val, y_val)

print(f"Train pairs: {len(train_pairs_1)}")
print(f"Val pairs: {len(val_pairs_1)}")

print(f"Kiara in train: {np.sum(y_train == 0)}")
print(f"Stanford in train: {np.sum(y_train == 1)}")

print("--- SIAMESE NEURAL NETWORK ARCHITECTURE ---")

def base_network(input_shape=(224,224,3)):
    input_tensor = Input(input_shape)
    x = tf.keras.applications.resnet50.preprocess_input(input_tensor)
    base_model = ResNet50(weights='imagenet', include_top=False)
    for i, layer in enumerate(base_model.layers):
        if i < len(base_model.layers) - 10:
            layer.trainable = False
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    model = Model(inputs=input_tensor, outputs=x)
    return model

def siamese_network(input_shape=(224,224,3)):
    input_1 = Input(input_shape)
    input_2 = Input(input_shape)
    base = base_network(input_shape)
    features_1 = base(input_1)
    features_2 = base(input_2)
    distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = distance([features_1, features_2])
    output = Dense(1, activation='sigmoid')(l1_distance)
    model = Model(inputs=[input_1, input_2], outputs=output)
    return model

model = siamese_network()
model.summary()

print("\n--- TRAINING THE SIAMESE NEURAL NETWORK ---")
optimizer = Adam(learning_rate=1e-4)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit([train_pairs_1, train_pairs_2], train_pair_labels, validation_data=([val_pairs_1, val_pairs_2], val_pair_labels), epochs=30, batch_size=32, callbacks=[early_stopping])
model.save('siamese_network.keras')

print("\n--- VALIDATION ---")
kiara_train_idx = np.where(y_train == 0)[0]
reference_idx = np.random.choice(kiara_train_idx, 5, replace=False)
reference_images = X_train[reference_idx]

predictions = []
for i in range(len(X_test)):
    scores = []
    for ref_img in reference_images:
        img_a = np.expand_dims(ref_img, 0)
        img_b = np.expand_dims(X_test[i], 0)
        similarity = model.predict([img_a, img_b], verbose=0)[0][0]
        scores.append(similarity)
    avg_score = np.mean(scores)
    if avg_score > 0.5:
        predictions.append(0)
    else:
        predictions.append(1)
predictions = np.array(predictions)

conf_matrix = confusion_matrix(y_test, predictions)
print(conf_matrix)
class_report = classification_report(y_test, predictions, target_names=['Kiara', 'Not Kiara'])
print(class_report)

groups = [
    (10, 'Kiara Standard'),
    (11, 'Kiara Young'),
    (12, 'Kiara Difficult'),
    (1, 'Similar'),
    (2, 'Normal'),
    (3, 'Different')
]
for value, name in groups:
    idx = np.where(y_test_detailed == value)[0]
    correct = np.sum(predictions[idx] == y_test[idx])
    accuracy = correct / len(idx)
    print(f"{name}: {accuracy*100:.2f}%")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(acc, label='accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.subplot(1, 2, 2)
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss, label='loss')
plt.plot(val_loss, label='validation loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('siamese_training_curves.jpg')
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Kiara', 'Nie Kiara'], yticklabels=['Kiara', 'Nie Kiara'])
plt.xlabel('Predykcja')
plt.ylabel('Prawda')
plt.title('Macierz pomyłek')
plt.savefig('siamese_confusion_matrix.jpg')
plt.show()

wrong_idx = np.where(predictions != y_test)[0]
print(f"Incorrect classifications: {len(wrong_idx)}")
plt.figure(figsize=(12, 4))
for i in range(len(wrong_idx)):
    plt.subplot(1, len(wrong_idx), i + 1)
    plt.imshow(X_test[wrong_idx[i]].astype(np.uint8))
    pred_label = "Kiara" if predictions[wrong_idx[i]] == 0 else "Nie Kiara"
    true_label = "Kiara" if y_test[wrong_idx[i]] == 0 else "Nie Kiara"
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}")
plt.savefig('siamese_test_prediction.jpg')
plt.show()