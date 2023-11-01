import argparse
from pathlib import Path

import pandas as pd

data_split_keys = {
    "train": "Training",
    "val": "Validating",
    "test": "Testing",
}


def save_keys(keys, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        for key in keys:
            f.write(key + "\n")
    print(f"Saved {len(keys)} keys to {save_path}")


def generate_keys(keys, FPBR_dir, dataset):
    data = (FPBR_dir.parent.parent.parent.parent / "dataset/PDBbind-v2020" / dataset / "data").glob("*")
    data = [d.name for d in data]
    data.sort()
    for i, split in enumerate(["train", "val", "test"]):
        docking_keys = [key for key in data if key.split("_")[0] in keys[i]]
        save_keys(docking_keys, FPBR_dir / dataset / f"{split}_keys.txt")


def generate_FPBR_keys(fpbr_csv_path):
    keys = get_keys_from_df(fpbr_csv_path)
    FPBR_dir = Path(__file__).parent
    generate_keys(keys, FPBR_dir, "scoring")
    generate_keys(keys, FPBR_dir, "docking")
    generate_keys(keys, FPBR_dir, "cross")
    generate_keys(keys, FPBR_dir, "pda")
    generate_keys(keys, FPBR_dir, "random")


def get_keys_from_df(fpbr_csv_path):
    df = pd.read_csv(fpbr_csv_path)
    df = df[["pdbid", "data_type"]]
    train_keys = df[df["data_type"] == data_split_keys["train"]]["pdbid"].tolist()
    val_keys = df[df["data_type"] == data_split_keys["val"]]["pdbid"].tolist()
    test_keys = df[df["data_type"] == data_split_keys["test"]]["pdbid"].tolist()
    train_keys.sort()
    val_keys.sort()
    test_keys.sort()
    return train_keys, val_keys, test_keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FPBR keys")
    parser.add_argument("--fpbr-csv-path", type=str, required=True, help="Path to FPBR dataset csv file")
    args = parser.parse_args()

    generate_FPBR_keys(args.fpbr_csv_path)
