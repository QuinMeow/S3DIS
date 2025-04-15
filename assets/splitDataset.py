'''
读取目录下所有CSV文件，并按比例分割训练、验证、测试集
'''
import os
import glob
import random
import argparse
import pandas as pd

def split_data(df, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    # 使用 pandas 随机打乱数据
    df = df.sample(frac=1, random_state=42)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]

def split_datasets(input_dir, output_dir):
    # 统计三个集合的数据
    train_all, val_all, test_all = [], [], []
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        train_df, val_df, test_df = split_data(df)
        train_all.append(train_df)
        val_all.append(val_df)
        test_all.append(test_df)

    # 拼接所有数据
    train_all_df = pd.concat(train_all, ignore_index=True) if train_all else pd.DataFrame()
    val_all_df = pd.concat(val_all, ignore_index=True) if val_all else pd.DataFrame()
    test_all_df = pd.concat(test_all, ignore_index=True) if test_all else pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    train_all_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_all_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_all_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

def split_train_and_val(csv_list, output_dir):
    # 统计训练集和验证集的数据
    train_all, val_all = [], []
    for csv_file in csv_list:
        df = pd.read_csv(csv_file)
        if df.empty:
            continue
        train_df, val_df, _ = split_data(df)
        train_all.append(train_df)
        val_all.append(val_df)

    # 拼接所有数据
    train_all_df = pd.concat(train_all, ignore_index=True) if train_all else pd.DataFrame()
    val_all_df = pd.concat(val_all, ignore_index=True) if val_all else pd.DataFrame()

    os.makedirs(output_dir, exist_ok=True)
    train_all_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_all_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)

if __name__ == "__main__":
    input_dir = "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset"
    output_dir = "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset"
    csv_list = ["area_1_pairs.csv", "area_2_pairs.csv", "area_4_pairs.csv", "area_5a_pairs.csv", "area_5b_pairs.csv", "area_6_pairs.csv"]
    split_train_and_val([os.path.join(input_dir, csv) for csv in csv_list], output_dir)
    # split_datasets(input_dir, output_dir)
