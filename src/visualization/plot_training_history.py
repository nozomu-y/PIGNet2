import argparse
import re
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="darkgrid")


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


def plot_training_history(df, save_path):
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # Plot training and testing scoring loss
    sns.lineplot(
        data=df,
        x="epoch",
        y="train_l_scoring",
        label="Training Set",
        ax=axes[0],
    )
    sns.lineplot(
        data=df,
        x="epoch",
        y="test_l_scoring",
        label="Test Set",
        ax=axes[0],
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Scoring Loss")

    # Plot training and testing Pearson correlation coefficient
    sns.lineplot(data=df, x="epoch", y="train_r", label="Training Set", ax=axes[1])
    sns.lineplot(data=df, x="epoch", y="test_r", label="Test Set", ax=axes[1])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Pearson Correlation")

    # Add titles
    axes[0].set_title("Scoring Loss")
    axes[1].set_title("Pearson Correlation Coefficient")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_log_path", type=str, help="Path to train log file")
    args = parser.parse_args()

    df = train_log_to_df(args.train_log_path)
    print(df)
    save_path = args.train_log_path.replace("train.log", "train_history.png")
    plot_training_history(df, save_path)
