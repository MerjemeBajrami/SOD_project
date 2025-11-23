#evaluate.py
import numpy as np
import tensorflow as tf



def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def batch_metrics(y_true, y_pred):
    y_pred_bin = tf.where(y_pred >= 0.5, 1.0, 0.0)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = iou_score(y_true, y_pred)
    return precision, recall, f1, iou

def evaluate_model(model, test_ds):
    precisions, recalls, f1s, ious = [], [], [], []
    for images_batch, masks_batch in test_ds:
        preds = model(images_batch, training=False)
        p, r, f, i = batch_metrics(masks_batch, preds)
        precisions.append(p.numpy())
        recalls.append(r.numpy())
        f1s.append(f.numpy())
        ious.append(i.numpy())

    print("Test results:")
    print(f"IoU: {np.mean(ious):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall: {np.mean(recalls):.4f}")
    print(f"F1: {np.mean(f1s):.4f}")
