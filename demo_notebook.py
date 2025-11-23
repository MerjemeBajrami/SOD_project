# demo_notebook.ipynb
import tensorflow as tf
from sod_model import build_unet_baseline,build_unet_bn_dropout
import cv2, matplotlib.pyplot as plt
import numpy as np
import os

# Load model
#model = build_unet_baseline()
model = build_unet_bn_dropout()

CHECKPOINT_DIR = "/Users/macair/Desktop/SOD/checkpoints_exps/bn_dropout/best_weights.weights.h5"
model.load_weights(CHECKPOINT_DIR)
def visualize_prediction(img_path, mask_path=None, model=model, IMG_SIZE=128):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    orig = tf.cast(img, tf.float32)/255.0
    h, w = orig.shape[:2]
    x = tf.expand_dims(tf.image.resize(orig, [IMG_SIZE, IMG_SIZE]),0)
    pred = model.predict(x)[0,...,0]
    pred_resized = cv2.resize(pred, (w,h))
    pred_bin = (pred_resized>=0.5).astype(np.uint8)

    plt.figure(figsize=(12,4))
    plt.subplot(1,4,1); plt.imshow(orig.numpy()); plt.title("Input"); plt.axis('off')
    if mask_path:
        m = tf.io.read_file(mask_path)
        m = tf.image.decode_image(m, channels=1)
        m = tf.cast(m, tf.float32)/255.0
        m = tf.image.resize(m,[h,w],method='nearest')
        plt.subplot(1,4,2); plt.imshow(m[:,:,0],cmap='gray'); plt.title("GT mask"); plt.axis('off')
    else:
        plt.subplot(1,4,2); plt.text(0.1,0.5,"No GT given",fontsize=12); plt.axis('off')
    plt.subplot(1,4,3); plt.imshow(pred_resized,cmap='gray'); plt.title("Pred mask"); plt.axis('off')
    overlay = orig.numpy().copy()
    overlay_mask = np.stack([pred_bin]*3,axis=-1)*np.array([1.0,0.2,0.2])
    combined = overlay*0.5 + overlay_mask*0.5
    plt.subplot(1,4,4); plt.imshow(combined); plt.title("Overlay"); plt.axis('off')
    plt.show()
# Demo image
demo_image = "/Users/macair/Desktop/SOD/ECSSD/images/0347.jpg"  # replace
demo_mask = "/Users/macair/Desktop/SOD/ECSSD/ground_truth_mask/0347.png"
visualize_prediction(demo_image, demo_mask, model=model)
