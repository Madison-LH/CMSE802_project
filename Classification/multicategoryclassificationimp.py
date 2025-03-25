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
    
    sampled_images = random.sample(all_images, min(88203, len(all_images)))
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
    ds = ds.map(preprocess, num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1024)
    ds = ds.cache()  # <-- moved AFTER .map()
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

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    preds = np.argmax(preds, axis=1)
    y_pred.extend(preds)
    y_true.extend(labels.numpy())


print(classification_report(y_true, y_pred, target_names=classes))
