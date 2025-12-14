import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/tables/results.csv")

for dataset in df["dataset"].unique():
    plt.figure()
    subset = df[df["dataset"] == dataset]

    for clf in subset["classifier"].unique():
        clf_data = subset[subset["classifier"] == clf]
        means = clf_data.groupby("train_fraction")["test_accuracy"].mean()
        plt.plot(means.index, means.values, marker="o", label=clf)

    plt.title(f"Test Accuracy vs Training Size ({dataset})")
    plt.xlabel("Training Fraction")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"results/figures/{dataset}_learning_curve.png")
    plt.close()
