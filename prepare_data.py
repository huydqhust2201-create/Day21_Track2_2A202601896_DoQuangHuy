"""
Tai va chuan bi tap du lieu Adult / Census Income (UCI Machine Learning Repository).

Khong can sua file nay. Chay: python prepare_data.py

Ket qua:
    data/train_batch1.csv   (22361 mau) - dung de huan luyen ngay
    data/holdout.csv        (500 mau)   - held-out set, khong bao gio huan luyen
    data/train_batch2.csv   (22361 mau) - danh cho phan huan luyen lien tuc (muc 4.13)
"""
import os
import urllib.request

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_DIR = "data/raw"
TRAIN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"

RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

FEATURE_COLUMNS = [
    "age", "workclass", "education_num", "marital_status", "occupation",
    "relationship", "sex", "capital_gain", "capital_loss", "hours_per_week",
]

CATEGORICAL_COLUMNS = ["workclass", "marital_status", "occupation", "relationship", "sex"]

RANDOM_SEED = 42
N_TRAIN_BATCH1 = 22361
N_HOLDOUT = 500
N_TRAIN_BATCH2 = 22361


def _download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Dang tai {url} ...")
    urllib.request.urlretrieve(url, dest)


def _load_raw() -> pd.DataFrame:
    train_path = os.path.join(RAW_DIR, "adult.data")
    test_path = os.path.join(RAW_DIR, "adult.test")
    _download(TRAIN_URL, train_path)
    _download(TEST_URL, test_path)

    df_train = pd.read_csv(
        train_path, header=None, names=RAW_COLUMNS,
        skipinitialspace=True, na_values="?",
    )
    # adult.test co 1 dong comment dau file, phai bo qua (skiprows=1)
    df_test = pd.read_csv(
        test_path, header=None, names=RAW_COLUMNS,
        skipinitialspace=True, na_values="?", skiprows=1,
    )
    # Nhan trong adult.test ket thuc bang dau "." (vi du "<=50K.")
    df_test["income"] = df_test["income"].str.rstrip(".")

    df = pd.concat([df_train, df_test], ignore_index=True)
    return df


def main() -> None:
    df = _load_raw()

    # Loai bo cac dong thieu gia tri (danh dau bang "?")
    df = df.dropna(axis=0, how="any").reset_index(drop=True)

    df["target"] = (df["income"] == ">50K").astype(int)

    out = df[FEATURE_COLUMNS + ["target"]].copy()

    # Ma hoa cac cot dang chuoi thanh so nguyen theo thu tu bang chu cai
    for col in CATEGORICAL_COLUMNS:
        out[col] = LabelEncoder().fit_transform(out[col])

    # Xao tron xac dinh (co the tai lap) truoc khi chia
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_idx = rng.permutation(len(out))
    out = out.iloc[shuffled_idx].reset_index(drop=True)

    n1, n2, n3 = N_TRAIN_BATCH1, N_HOLDOUT, N_TRAIN_BATCH2
    train_batch1 = out.iloc[:n1]
    holdout = out.iloc[n1:n1 + n2]
    train_batch2 = out.iloc[n1 + n2:n1 + n2 + n3]

    os.makedirs("data", exist_ok=True)
    train_batch1.to_csv("data/train_batch1.csv", index=False)
    holdout.to_csv("data/holdout.csv", index=False)
    train_batch2.to_csv("data/train_batch2.csv", index=False)

    pos_rate = out["target"].mean() * 100

    print(f"train_batch1.csv : {len(train_batch1)} mau")
    print(f"holdout.csv      : {len(holdout)} mau")
    print(f"train_batch2.csv : {len(train_batch2)} mau")
    print(f"Ty le lop >50K   : {pos_rate:.1f}%")


if __name__ == "__main__":
    main()
