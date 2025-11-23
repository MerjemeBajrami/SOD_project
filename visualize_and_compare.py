
import os, json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CHECKPOINT_ROOT = "./checkpoints_exps"
exps = ['baseline','bn_dropout']
table_rows = []
plt.figure(figsize=(10,4))

for idx, e in enumerate(exps):
    ckpt_dir = os.path.join(CHECKPOINT_ROOT, e)
    hist_csv = os.path.join(ckpt_dir, "history.csv")
    test_json = os.path.join(ckpt_dir, "test_results.json")
    if os.path.exists(test_json):
        with open(test_json) as f:
            tr = json.load(f)
    else:
        tr = {'IoU':None,'Precision':None,'Recall':None,'F1':None}
    table_rows.append([e, tr['IoU'], tr['Precision'], tr['Recall'], tr['F1']])
    if os.path.exists(hist_csv):
        df = pd.read_csv(hist_csv)
        plt.plot(df['epoch'], df['val_f1'], label=f"{e} val_f1")
plt.xlabel("Epoch"); plt.ylabel("Val F1"); plt.legend(); plt.title("Validation F1 per epoch")
plt.tight_layout()
plt.savefig("val_f1_comparison.png", dpi=150)
plt.show()

df_table = pd.DataFrame(table_rows, columns=['experiment','IoU','Precision','Recall','F1'])
print(df_table.to_markdown(index=False))
df_table.to_csv("experiments_summary.csv", index=False)
