from data_loader import load_datasets
from sod_model import build_unet_baseline
from evaluate import evaluate_model

# Load datasets
_, _, test_ds = load_datasets()

# Build model
model = build_unet_baseline()

# Load trained weights
model.load_weights("checkpoints/best_weights.weights.h5")

# Evaluate
evaluate_model(model, test_ds)

