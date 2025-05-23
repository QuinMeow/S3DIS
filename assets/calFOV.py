'''
计算视场角分布
'''
import os
import json
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def get_field_of_view_rads_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data.get('field_of_view_rads')

def calculate_statistics(directory):
    field_of_view_rads_values = []

    # 获取所有 JSON 文件的路径
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))

    # 使用 tqdm 显示进度条
    for file_path in tqdm.tqdm(json_files, desc="Processing JSON files"):
        fov_rads = get_field_of_view_rads_from_json(file_path)
        if fov_rads is not None:
            field_of_view_rads_values.append(fov_rads)

    if not field_of_view_rads_values:
        print("No 'field_of_view_rads' values found in the directory.")
        return

    average_fov = np.mean(field_of_view_rads_values)
    distribution = np.histogram(field_of_view_rads_values, bins='auto')

    print(f"Average 'field_of_view_rads': {average_fov}")
    print(f"Distribution of 'field_of_view_rads': {distribution}")

    # 绘制分布曲线图
    plt.figure(figsize=(10, 6))
    plt.hist(field_of_view_rads_values, bins="auto", alpha=0.7, rwidth=0.85)
    plt.title("Distribution of field_of_view_rads")
    plt.xlabel("field_of_view_rads")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()

if __name__ == "__main__":
    directory = "/data2/zyz/S3DIS/area_3/pano/pose"
    calculate_statistics(directory)
