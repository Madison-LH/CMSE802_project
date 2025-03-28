"""
Module for model implementation and classification

This module contains functions for loading the dataset,
train/test/validation splitting, model training, and
classification

Author: Madison Honore
Date: March 2025
"""


import os
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import os



data_roots = [r"E:\datasetResized"]  # Add as needed
classes = ['ENTR', 'EXIT', 'INSC', 'NOBR', 'PERC']  # behaviors

# Step 1: Collect image paths and labels
image_paths = []
labels = []

for cls in classes:
    all_images = []
    for root in data_roots:
        cls_path = os.path.join(root, cls)
        if os.path.exists(cls_path):
            all_images += [os.path.join(cls_path, img) for img in os.listdir(cls_path)
                           if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    sampled_images = random.sample(all_images, min(25000, len(all_images)))
    image_paths.extend(sampled_images)
    labels.extend([cls] * len(sampled_images))

# Step 2: Encode labels to integers
class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
y = [class_to_idx[lbl] for lbl in labels]

# Step 3: Use sklearn for splitting
X_train, X_temp, y_train, y_temp = train_test_split(image_paths, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

IMG_SIZE = (224, 224)
batch_size = 64
AUTOTUNE = tf.data.AUTOTUNE

def preprocess(img_path, label):
    img = tf.io.read_file(img_path)  # img is tf.string -> becomes tf.uint8 after decode_jpeg
    img = tf.image.decode_jpeg(img, channels=3)  # tf.uint8 image tensor now
    img = tf.cast(img, tf.float32)  # must cast BEFORE preprocess_input()
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # now tf.float32 [-1,1]
    return img, tf.cast(label, tf.int32)  # force label to tf.int32

def create_tf_dataset(paths, labels, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(preprocess, num_parallel_calls=2)
    ds = ds.apply(tf.data.experimental.ignore_errors())    
    if shuffle:
        ds = ds.shuffle(1024)
    #ds = ds.cache()  # <-- moved AFTER .map()
    return ds.batch(batch_size).prefetch(AUTOTUNE)

train_ds = create_tf_dataset(X_train, y_train, shuffle=True)
val_ds = create_tf_dataset(X_val, y_val)
test_ds = create_tf_dataset(X_test, y_test)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
base_model.trainable = False  # Initially freeze

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(classes), activation='softmax')
])

# ---- COMPILE + TRAIN SECTION ----

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # freeze training stage
)

model.save("mobilenet_behavior_classifier.keras")  # SavedModel or Keras HDF5 format

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())


print(classification_report(y_true, y_pred, target_names=classes))


# ---- VISUALIZATIONS ----

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot it
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, zero_division=0)

x = np.arange(len(classes))
plt.figure(figsize=(10, 6))
plt.bar(x - 0.2, precision, 0.2, label='Precision')
plt.bar(x, recall, 0.2, label='Recall')
plt.bar(x + 0.2, f1, 0.2, label='F1 Score')
plt.xticks(x, classes)
plt.ylabel('Score')
plt.title('Per-Class Metrics')
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

# Only works if you stored predicted probabilities from model.predict()
# For example: probs = model.predict(images) ← store this during prediction loop

# Convert labels to one-hot
y_true_bin = label_binarize(y_true, classes=range(len(classes)))

# Plot ROC for each class
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (One-vs-All)')
plt.legend()
plt.show()

import seaborn as sns

confidences = np.max(all_probs, axis=1)  # max softmax probability
sns.histplot(confidences, bins=20, kde=True)
plt.title('Model Confidence Distribution')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.show()


import matplotlib.pyplot as plt
wrong_idx = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
sample_wrong = random.sample(wrong_idx, 16)

plt.figure(figsize=(12, 12))
for i, idx in enumerate(sample_wrong):
    img = tf.io.read_file(X_test[idx])
    img = tf.image.decode_jpeg(img, channels=3)
    plt.subplot(4, 4, i + 1)
    plt.imshow(img.numpy().astype("uint8"))
    plt.title(f'True: {classes[y_true[idx]]}\nPred: {classes[y_pred[idx]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
