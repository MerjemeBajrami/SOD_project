# data_loader.py
import tensorflow as tf
import glob, os
from sklearn.model_selection import train_test_split

IMG_SIZE = 128
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
SEED = 42

def get_pairs(images_dir, masks_dir):
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*")))
    mask_paths = []
    for p in img_paths:
        name = os.path.splitext(os.path.basename(p))[0]
        found = None
        for ext in [".png", ".bmp", ".jpg", ".jpeg", ".tif"]:
            cand = os.path.join(masks_dir, name + ext)
            if os.path.exists(cand):
                found = cand
                break
        if found:
            mask_paths.append(found)
        else:
            matches = [m for m in glob.glob(os.path.join(masks_dir, "*")) if name in os.path.basename(m)]
            mask_paths.append(matches[0] if matches else None)
    paired = [(i, m) for i, m in zip(img_paths, mask_paths) if m is not None]
    images, masks = zip(*paired)
    return list(images), list(masks)

def parse_image(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0

    m = tf.io.read_file(mask_path)
    m = tf.image.decode_image(m, channels=1)
    m.set_shape([None, None, 1])
    m = tf.image.resize(m, [IMG_SIZE, IMG_SIZE], method='nearest')
    m = tf.cast(m, tf.float32) / 255.0
    m = tf.where(m >= 0.5, 1.0, 0.0)
    return img, m
def augment(img, mask):
    # flip
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    # rotate 0/90/180/270
    if tf.random.uniform(()) > 0.8:
        k = tf.random.uniform((), 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        mask = tf.image.rot90(mask, k)
    # brightness
    if tf.random.uniform(()) > 0.7:
        img = tf.image.random_brightness(img, 0.15)
        img = tf.clip_by_value(img, 0.0, 1.0)
    # contrast
    if tf.random.uniform(()) > 0.75:
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
    # gaussian noise
    if tf.random.uniform(()) > 0.85:
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=0.02)
        img = img + noise
        img = tf.clip_by_value(img, 0.0, 1.0)
    return img, mask

def build_dataset(img_paths, mask_paths, batch=BATCH_SIZE, shuffle=True, training=True):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(len(img_paths), seed=SEED)
    ds = ds.map(lambda x, y: tf.py_function(parse_image, [x, y], [tf.float32, tf.float32]), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda a, b: (tf.ensure_shape(a, [IMG_SIZE, IMG_SIZE, 3]), tf.ensure_shape(b, [IMG_SIZE, IMG_SIZE, 1])), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

def get_splits(images_dir, masks_dir, test_size=0.3, val_ratio=0.5):
    images, masks = get_pairs(images_dir, masks_dir)
    train_imgs, rest_imgs, train_masks, rest_masks = train_test_split(images, masks, test_size=test_size, random_state=SEED)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(rest_imgs, rest_masks, test_size=val_ratio, random_state=SEED)
    return (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)
def load_datasets(img_size=128, batch_size=32, seed=42):
    """
    Returns train, val, test tf.data.Dataset objects.
    """
    from sklearn.model_selection import train_test_split
    import os, glob, tensorflow as tf

    BASE = "./ECSSD"  # adjust path to your local ECSSD folder
    IMAGES_DIR = os.path.join(BASE, "images")
    MASKS_DIR = os.path.join(BASE, "ground_truth_mask")

    # reuse your get_pairs, parse_image, augment, build_dataset functions
    images, masks = get_pairs(IMAGES_DIR, MASKS_DIR)

    train_imgs, rest_imgs, train_masks, rest_masks = train_test_split(images, masks, test_size=0.30, random_state=seed)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(rest_imgs, rest_masks, test_size=0.5, random_state=seed)

    train_ds = build_dataset(train_imgs, train_masks, batch=batch_size, shuffle=True, training=True)
    val_ds = build_dataset(val_imgs, val_masks, batch=batch_size, shuffle=False, training=False)
    test_ds = build_dataset(test_imgs, test_masks, batch=batch_size, shuffle=False, training=False)

    return train_ds, val_ds, test_ds
