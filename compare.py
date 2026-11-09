import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv").set_index("run")
folds = [c for c in runs.columns if c.startswith("fold_")]

with_selection = runs.loc["20261109_114705", folds]
without = runs.loc["20261109_114848", folds]

print(pd.DataFrame({"selection on": with_selection, "selection off": without}))
print(corrected_paired_test(without, with_selection, test_train_ratio=1 / 4))