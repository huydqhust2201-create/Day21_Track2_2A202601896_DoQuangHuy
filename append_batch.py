"""
Mo phong viec thu thap them du lieu: noi train_batch2.csv vao train_batch1.csv.

Khong can sua file nay. Chay: python append_batch.py
"""
import pandas as pd

BATCH1_PATH = "data/train_batch1.csv"
BATCH2_PATH = "data/train_batch2.csv"


def main() -> None:
    df1 = pd.read_csv(BATCH1_PATH)
    df2 = pd.read_csv(BATCH2_PATH)

    n_before = len(df1)
    combined = pd.concat([df1, df2], ignore_index=True)
    combined.to_csv(BATCH1_PATH, index=False)
    n_after = len(combined)

    print(f"Cap nhat du lieu: {n_before} -> {n_after} mau")


if __name__ == "__main__":
    main()
