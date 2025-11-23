# train_experiments.py
import os, time, csv, numpy as np
import tensorflow as tf
from data_loader import get_splits, build_dataset, IMG_SIZE
from sod_model import get_model_by_name, sod_loss, iou_score

# ---------------- config ----------------
BASE = "./ECSSD"
IMAGES_DIR = BASE + "/images"
MASKS_DIR = BASE + "/ground_truth_mask"
CHECKPOINT_ROOT = "./checkpoints_exps"
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

BATCH_SIZE = 16  # reduce if OOM
EPOCHS = 25
SEED = 42
LR = 1e-3

# ---------------- helpers ----------------
def batch_metrics(y_true, y_pred):
    y_pred_bin = tf.where(y_pred>=0.5, 1.0, 0.0)
    y_true_f = tf.reshape(y_true, [-1]); y_pred_f = tf.reshape(y_pred_bin, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    precision = tp/(tp+fp+1e-8)
    recall = tp/(tp+fn+1e-8)
    f1 = 2*precision*recall/(precision+recall+1e-8)
    iou = iou_score(y_true, y_pred)
    return precision, recall, f1, iou

def run_experiment(model_name, lr=LR, batch_size=BATCH_SIZE, epochs=EPOCHS):
    print("Running:", model_name)
    # dataset splits
    (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks) = get_splits(IMAGES_DIR, MASKS_DIR)
    train_ds = build_dataset(train_imgs, train_masks, batch=batch_size, shuffle=True, training=True)
    val_ds = build_dataset(val_imgs, val_masks, batch=batch_size, shuffle=False, training=False)
    test_ds = build_dataset(test_imgs, test_masks, batch=batch_size, shuffle=False, training=False)

    # model
    model = get_model_by_name(model_name)
    optimizer = tf.keras.optimizers.Adam(lr)

    # lr scheduler example (use ReduceLROnPlateau)
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    ckpt_dir = os.path.join(CHECKPOINT_ROOT, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = os.path.join(ckpt_dir, "best_weights.weights.h5")

    best_val_loss = 1e9; patience = 6; wait = 0

    # optional: compile for convenience (not necessary with manual train loop)
    model.compile(optimizer=optimizer, loss=sod_loss)

    history = {'epoch':[], 'train_loss':[], 'val_loss':[], 'val_iou':[], 'val_f1':[]}
    for epoch in range(1, epochs+1):
        t0 = time.time()
        # train loop (use fit for simplicity with our custom loss)
        hist = model.fit(train_ds, validation_data=val_ds, epochs=1, callbacks=[reduce_on_plateau], verbose=1)
        train_loss = hist.history['loss'][-1]
        val_loss = hist.history['val_loss'][-1] if 'val_loss' in hist.history else 0.0

        # compute metrics on validation set
        precisions=[]; recalls=[]; f1s=[]; ious=[]
        for images_batch, masks_batch in val_ds:
            preds_val = model.predict(images_batch)
            p,r,f,i = batch_metrics(masks_batch, tf.convert_to_tensor(preds_val))
            precisions.append(p.numpy()); recalls.append(r.numpy()); f1s.append(f.numpy()); ious.append(i.numpy())
        mean_f1 = np.mean(f1s); mean_iou = np.mean(ious)

        print(f"Epoch {epoch}/{epochs} - t_loss: {train_loss:.4f} - v_loss: {val_loss:.4f} - val_iou: {mean_iou:.4f} - val_f1: {mean_f1:.4f} - time: {time.time()-t0:.1f}s")

        history['epoch'].append(epoch); history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['val_iou'].append(mean_iou); history['val_f1'].append(mean_f1)

        # save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss; wait = 0
            model.save_weights(checkpoint_path)
            print("Saved best weights to", checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    # final test evaluation using saved best weights
    model.load_weights(checkpoint_path)
    # evaluate on test set
    precisions, recalls, f1s, ious = [], [], [], []
    for images_batch, masks_batch in test_ds:
        preds = model.predict(images_batch)
        p,r,f,i = batch_metrics(masks_batch, tf.convert_to_tensor(preds))
        precisions.append(p.numpy()); recalls.append(r.numpy()); f1s.append(f.numpy()); ious.append(i.numpy())
    test_results = {'IoU': float(np.mean(ious)), 'Precision': float(np.mean(precisions)), 'Recall': float(np.mean(recalls)), 'F1': float(np.mean(f1s))}
    print("Test results:", test_results)

    # save history and test_results to CSV
    csv_path = os.path.join(ckpt_dir, "history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss','val_loss','val_iou','val_f1'])
        for i in range(len(history['epoch'])):
            writer.writerow([history['epoch'][i], history['train_loss'][i], history['val_loss'][i], history['val_iou'][i], history['val_f1'][i]])
    # save test results
    tr_path = os.path.join(ckpt_dir, "test_results.json")
    import json
    with open(tr_path, 'w') as f:
        json.dump(test_results, f, indent=2)

    return test_results, ckpt_dir

if __name__ == "__main__":
    # run baseline and two variants (one by one). comment/uncomment as needed.
    exps = ['baseline', 'bn_dropout', 'deep']  # baseline, Variant A, Variant B
    results = {}
    for e in exps:
        res, ckpt_dir = run_experiment(e)
        results[e] = res
    print("All experiment results:")
    print(results)
