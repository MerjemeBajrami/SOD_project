# 🧠 Salient Object Detection (SOD) — Deep Learning Project

This project implements a **Salient Object Detection (SOD)** model using TensorFlow/Keras.  
The goal is to train a neural network that identifies the **most visually important region** in an image by generating a corresponding saliency mask.

The project includes:

- 📦 Dataset loading & preprocessing  
- 🧱 Model architecture (U-Net + improvements)  
- 🏋️ Training pipeline with checkpoints  
- 📊 Evaluation (IoU, Precision, Recall, F1, MAE)  
- 🖼️ Demo notebook for running predictions  

---

## 📂 Project Structure

SOD/
│── checkpoints/               # Baseline model checkpoints
│── checkpoints_exps/          # Experiment model checkpoints
│── ECSSD/                     # Dataset (images + masks)
│── venv / venv_new / venv_tf  # Virtual environments (ignored via .gitignore)
│
│── data_loader.py             # Dataset loading + augmentations
│── sod_model.py               # Baseline & improved U-Net architectures
│── train.py                   # Training script
│── evaluate.py                # Evaluation script (IoU, F1, Precision, Recall)
│── visualize_and_compare.py   # Visualize GT vs baseline vs improved model
│── run_eval.py                # Quick eval runner
│── demo_notebook.py           # Demo script for predictions
│
│── experiments_summary.csv    # Table comparing baseline & improved models
│── val_f1_comparison.png      # Plot comparing validation F1 curves
│── requirements.txt           # Dependencies
│── README.md                  # This file
│── .gitignore




---

## 📁 Dataset

Supported datasets:

- **ECSSD**
- **DUTS**
- **HKU-IS**

Expected folder structure:

ECSSD/
├── images/
└── ground_truth_mask/

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```


## Run the training script:

python train.py


This will:

Load the dataset

Split into train/validation/test

Train the U-Net model

Save the best model weights in checkpoints/



## Metrics:

IoU

Precision

Recall

F1-Score
##  Demo (Visualization)

Open:

demo_notebook.ipynb


Inside, you can:

Load a sample image

Predict its saliency mask

Visualize input, ground truth, prediction, and overlay

🚀 Features
✔️ Baseline U-Net

4 encoder blocks + bottleneck + 4 decoder blocks

Loss = BCE + α·(1 – IoU)

Metrics: IoU, Precision, Recall, F1-score

## ✔️ Dataset Pipeline

Auto-pairing images & masks

Augmentations:

Random horizontal flip

Random brightness

Random rotation

tf.data with caching, batching, prefetching

## ✔️ Experiments Included

Two improvement experiments were run:

Experiment 1 — Add Dropout + BatchNorm

Improves generalization

Stabilizes training

## The best weights are saved automatically:

checkpoints/best_weights.weights.h5


## Improved model weights:

checkpoints_exps/best_weights_exp1.h5
checkpoints_exps/best_weights_exp2.h5

## 🧪 Evaluation

Run on test set:

python run_eval.py
