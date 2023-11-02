import argparse
import re
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")

metric_to_label = {
    "l_scoring": "Scoring Loss",
    "l_docking": "Docking Loss",
    "l_cross": "Cross Docking Loss",
    "l_random": "Random Docking Loss",
    "l_pda": "Positive Data Augmentation Loss",
    "l_dvdw": "VDW Loss",
    "r": "Pearson Correlation",
    "tau": "Kendall Tau",
}


def extract_table_from_train_log(train_log_path):
    with open(train_log_path, "r") as f:
        lines = f.readlines()
    lines = [
        line
        for line in lines
        if re.fullmatch(r"\d+\s+[\d\-\.\s]+", line) != None or line.startswith("epoch")
    ]
    table_text = "".join(lines)
    return table_text


def train_log_to_df(train_log_path):
    table_text = extract_table_from_train_log(train_log_path)
    io = StringIO(table_text)
    df = pd.read_table(io, sep="\t")
    return df


def plot_training_history(df, metric, save_dir):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot training history for the given metric
    sns.lineplot(
        data=df,
        x="epoch",
        y="train_" + metric,
        label="Training Set",
        ax=ax,
        linewidth=0.8,
    )
    sns.lineplot(
        data=df,
        x="epoch",
        y="val_" + metric,
        label="Validation Set",
        ax=ax,
        linewidth=0.8,
    )
    sns.lineplot(
        data=df,
        x="epoch",
        y="test_" + metric,
        label="Test Set",
        ax=ax,
        linewidth=0.8,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_to_label[metric])
    ax.set_title(metric_to_label[metric])

    plt.tight_layout()
    save_path = save_dir + "/training_history_" + metric + ".png"
    plt.savefig(save_path, dpi=300)


def plot_training_histories(df, save_dir):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    metrics = list(df.columns[1:])
    metrics = [metric[6:] for metric in metrics if metric.startswith("train")]
    for metric in metrics:
        print("Plotting training history for " + metric)
        plot_training_history(df, metric, save_dir)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "training_output_dir", type=str, help="Path to training output directory"
    )
    args = parser.parse_args()

    train_log_path = args.training_output_dir + "/train.log"
    df = train_log_to_df(train_log_path)
    save_dir = args.training_output_dir + "/figures"
    plot_training_histories(df, save_dir)
