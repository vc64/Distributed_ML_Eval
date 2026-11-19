import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

if __package__:
    from .model import train_forest
else:
    from model import train_forest


def load_data(data_dir):
    data_dir = Path(data_dir)
    train_data = pd.read_parquet(data_dir / "train_data.parquet")
    test_data = pd.read_parquet(data_dir / "test_data.parquet")
    return (
        train_data.drop(columns="label").to_numpy(),
        test_data.drop(columns="label").to_numpy(),
        train_data["label"].to_numpy(),
        test_data["label"].to_numpy(),
    )


def main():
    parser = argparse.ArgumentParser(description="Train a Covertype random-forest baseline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--trees", type=int, default=50)
    args = parser.parse_args()

    if args.trees < 1:
        parser.error("--trees must be positive")

    features_train, features_test, labels_train, labels_test = load_data(args.data_dir)

    started_at = time.perf_counter()
    forest = train_forest(features_train, labels_train, n_estimators=args.trees)
    training_seconds = time.perf_counter() - started_at

    predictions = forest.predict(features_test)
    print(f"train_rows: {len(labels_train)}")
    print(f"test_rows: {len(labels_test)}")
    print(f"trees: {args.trees}")
    print(f"training_seconds: {training_seconds:.3f}")
    print(f"accuracy: {accuracy_score(labels_test, predictions):.4f}")
    print(f"macro_f1: {f1_score(labels_test, predictions, average='macro'):.4f}")


if __name__ == "__main__":
    main()
