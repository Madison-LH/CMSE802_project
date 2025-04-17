
"""
Module for dataset augmentation/balancing

This module contains functions for augmenting
and balancing the the imbalanced classes with
synthetic data (rotations, flips, inversions, etc)

Author: Madison Honore
Date: March 2025
"""


import os
import numpy as np
import random
import multiprocessing
from skimage import io, util
from skimage.exposure import rescale_intensity
from skimage.transform import rotate
from tqdm import tqdm


def augment_and_balance_dataset(input_dir, output_dir, target_class_balance=88203):
    """
    Augments an image dataset while ensuring images are randomly sampled to avoid overrepresentation.
    Guarantees each class reaches the target size.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect class distributions
    class_counts = {}
    for class_label in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_label)
        if os.path.isdir(class_path):
            class_counts[class_label] = len(os.listdir(class_path))

    # Determine target class size
    max_class_size = max(class_counts.values())
    target_size = target_class_balance if target_class_balance else max_class_size
    print(f"🔍 Target class size per category: {target_size}")

    tasks = []
    for class_label, count in class_counts.items():
        if count >= target_size:
            continue  # Skip classes that are already balanced

        class_input_path = os.path.join(input_dir, class_label)
        class_output_path = os.path.join(output_dir, class_label)
        os.makedirs(class_output_path, exist_ok=True)

        image_files = [f for f in os.listdir(
            class_input_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        if not image_files:
            print(f"⚠️ Warning: No images found in {class_input_path}")
            continue

        images = {img_file: os.path.join(
            class_input_path, img_file) for img_file in image_files}

        # Keep augmenting until target size is reached
        while count < target_size:
            image_sample = random.sample(image_files, min(
                len(image_files), target_size - count))

            for img_name in image_sample:
                img_path = images[img_name]
                tasks.append((img_path, img_name, class_output_path))
                count += 1

    if not tasks:
        print("✅ No augmentation needed. Dataset already balanced.")
        return

    print(
        f"🚀 Augmenting {len(tasks)} images using {min(multiprocessing.cpu_count(), 8)} workers...")

    # Use multiprocessing with tqdm for progress tracking
    with multiprocessing.Pool(processes=min(multiprocessing.cpu_count(), 8)) as pool:
        for _ in tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processing Images"):
            pass

    print("🎉 Augmentation completed!")


def process_image(task):
    """
    Loads an image, applies random augmentation, and saves it with a unique filename.
    """
    try:
        img_path, img_name, class_output_path = task
        image = io.imread(img_path)

        if image is None or image.size == 0:
            print(f"⚠️ Error: Image {img_path} could not be loaded!")
            return

        augmented_img, aug_desc = random_augment(image)

        # Generate new filename
        base_name, ext = os.path.splitext(img_name)
        new_filename = f"{base_name}_{aug_desc}{ext}"
        save_path = os.path.join(class_output_path, new_filename)

        # Ensure filename uniqueness
        counter = 1
        while os.path.exists(save_path):
            new_filename = f"{base_name}_{aug_desc}_{counter}{ext}"
            save_path = os.path.join(class_output_path, new_filename)
            counter += 1

        io.imsave(save_path, augmented_img)

    except Exception as e:
        print(f"❌ Error processing {img_path}: {e}")


def random_augment(image):
    """
    Applies a random augmentation to an image while ensuring it remains readable.
    """
    aug_desc = []

    # Ensure image is in correct dtype
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Convert to uint8

    # Flip horizontally
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
        aug_desc.append("flipped")

    # Flip vertically
    if np.random.rand() > 0.5:
        image = np.flipud(image)
        aug_desc.append("flippedV")

    # Rotate image (smaller range to prevent distortions)
    if np.random.rand() > 0.5:
        angle = np.random.uniform(-15, 15)  # More conservative rotation range
        image = rotate(image, angle=angle, mode='reflect',
                       preserve_range=True).astype(np.uint8)
        aug_desc.append(f"rotated{int(angle)}")

    # Rescale intensity safely (only if contrast is too low)
    min_val, max_val = image.min(), image.max()
    if np.random.rand() > 0.5 and max_val - min_val < 50:  # Require significant contrast difference
        image = image.astype(np.float32) / 255.0  # Normalize
        image = rescale_intensity(image, in_range=(
            min_val, max_val), out_range=(0, 1))
        image = (image * 255).astype(np.uint8)
        aug_desc.append("rescaled")

    # Invert colors only if the image is not already bright
    mean_pixel = np.mean(image)
    if np.random.rand() > 0.3 and mean_pixel < 150:  # Skip inversion for bright images
        if len(image.shape) == 3 and image.shape[2] == 3:  # RGB
            image = util.invert(image)
        elif len(image.shape) == 2:  # Grayscale
            image = 255 - image
        aug_desc.append("inverted")

    # **Final safeguard: Fix extremely dark images**
    if mean_pixel < 20:  # If the image is too dark, apply contrast stretch
        image = rescale_intensity(image, in_range=(
            0, 40), out_range=(0, 255))  # Stretch dark values
        aug_desc.append("contrast_stretched")

    aug_desc = "_".join(aug_desc) if aug_desc else "original"
    return np.clip(image, 0, 255).astype(np.uint8), aug_desc


if __name__ == "__main__":
    augment_and_balance_dataset(
        r"E:\Multicategory_frames", r"E:\Augmented_frames")
