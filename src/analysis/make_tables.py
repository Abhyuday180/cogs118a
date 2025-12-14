import pandas as pd

df = pd.read_csv("results/tables/results.csv")

summary = (
    df
    .groupby(["dataset", "classifier", "train_fraction"])
    ["test_accuracy"]
    .agg(["mean", "std"])
    .reset_index()
)

summary.to_csv("results/tables/summary_table.csv", index=False)
print(summary)
