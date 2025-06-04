'''
从S3DIS原有的RGB图像中，提取样本对
'''
import os
import json
import shutil
import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from math import pi

# 设置数据路径
dataset_name = "area_6"
data_path = f"/data2/zyz/S3DIS/{dataset_name}/data"
rgb_path = os.path.join(data_path, "rgb")
pose_path = os.path.join(data_path, "pose")
pickup_path = "/data2/zyz/S3DIS/ImagePairs/180_7classes"
csv_name = f"{dataset_name}_pairs.csv"

# 确保 pickUp 目录存在
os.makedirs(pickup_path, exist_ok=True)

# 解析图片和姿态数据
image_data = []
pose_data = {}

# 遍历 pose 文件夹，读取 json 数据
for pose_file in tqdm(os.listdir(pose_path), desc="Processing pose files"):
    if pose_file.endswith(".json"):
        pose_filepath = os.path.join(pose_path, pose_file)
        with open(pose_filepath, "r") as f:
            data = json.load(f)
            pose_data[pose_file] = data

# 遍历 rgb 文件夹，匹配对应的 pose 数据
for img_file in tqdm(os.listdir(rgb_path), desc="Processing image files"):
    if img_file.endswith(".png"):
        parts = img_file.split("_")
        camera_uuid = parts[1]
        room = parts[2] + "_" + parts[3]
        frame_num = int(parts[5])
        pose_file = f"camera_{camera_uuid}_{room}_frame_{frame_num}_domain_pose.json" # 文件名中只有房间的前两项，根据UUID区别不同的拍摄位置
        
        if pose_file in pose_data:
            data = pose_data[pose_file]
            image_data.append({
                "image_file": img_file,
                "room": data["room"], # json文件中的room包含area编号
                "camera_uuid": camera_uuid,
                "camera_location": data["camera_location"],
                "frame_num": frame_num,
                "final_camera_rotation": data["final_camera_rotation"],
                "fx": data["camera_k_matrix"][0][0],
                "fy": data["camera_k_matrix"][1][1]
            })

# 转换为 pandas DataFrame
df = pd.DataFrame(image_data)

# 分组并提取图片对
pairs = []

# 8个方向
directions_360_8classes = [
        ("F", -22.5, 22.5),
        ("FR", -67.5, -22.5),
        ("R", -112.5, -67.5),
        ("BR", -157.5, -112.5),
        ("B", 157.5, -157.5),
        ("BL", 112.5, 157.5),
        ("L", 67.5, 112.5),
        ("FL", 22.5, 67.5),
    ]

# 180°内7方向
directions_180_7classes = [
        ("0°", -15, 15),
        ("30°", 15, 45),
        ("60°", 45, 75),
        ("90°", 75, 105),
        ("-30°", -45, -15),
        ("-60°", -75, -45),
        ("-90°", -105, -75),
    ]

# 先按room分组，然后按camera_uuid进一步分组
for room, room_group in tqdm(df.groupby("room"), desc="Processing rooms"):
    for camera_uuid, group in room_group.groupby("camera_uuid"):
        # 重置索引以确保索引连续
        group = group.reset_index(drop=True)
        
        # 手动创建所有可能的配对
        all_pairs = []
        for i in range(len(group)):
            for j in range(i+1, len(group)):  # 避免重复配对和自身配对
                all_pairs.append((i, j))
        
        # 随机打乱配对顺序
        np.random.shuffle(all_pairs)
        
        valid_pairs = []
        # 追踪每个角度范围内已找到的配对数量
        # yaw_range_counts = {(30, 60): 0, (60, 90): 0, (90, 120): 0, (120, 150): 0, (150, 180): 0}
        yaw_range_counts = {(0, 15): 0, (15, 45): 0, (45, 75): 0, (75, 105): 0}
        
        # 遍历打乱后的配对
        for idx_a, idx_b in all_pairs:
            img_a = group.iloc[idx_a]
            img_b = group.iloc[idx_b]
            
            pitch_a, roll_a, yaw_a = img_a["final_camera_rotation"]
            pitch_b, roll_b, yaw_b = img_b["final_camera_rotation"]
            
            pitch_diff = (pitch_b - pitch_a) / pi * 180
            yaw_diff = (yaw_b - yaw_a) / pi * 180
            yaw_diff = (yaw_diff + 180) % 360 - 180 # 修正到 -180 ~ 180 之间
            
            if abs(pitch_diff) <= 30:
                # for yaw_range in [(30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]:
                for yaw_range in [(0, 15), (15, 45), (45, 75), (75, 105)]:  # 180°内7方向
                    if yaw_range[0] <= abs(yaw_diff) < yaw_range[1] and yaw_range_counts[yaw_range] < 5:
                        # for direction, lower, upper in directions_360_8classes:
                        for direction, lower, upper in directions_180_7classes:  # 180°内7方向
                            if lower <= yaw_diff < upper or (lower > upper and (yaw_diff >= lower or yaw_diff < upper)):
                                b2a_direction = direction
                        valid_pairs.append({
                            "room": room,
                            "camera_uuid": camera_uuid,
                            "camera_location": img_a["camera_location"],
                            "frame_num_a": img_a["frame_num"],
                            "frame_num_b": img_b["frame_num"],
                            "yaw_diff": yaw_diff,
                            "b2a_direction" : b2a_direction,
                            "pitch_diff": pitch_diff,
                            "fx_a": img_a["fx"],
                            "fy_a": img_a["fy"],
                            "fx_b": img_b["fx"],
                            "fy_b": img_b["fy"],
                            "final_camera_rotation_a": img_a["final_camera_rotation"],
                            "final_camera_rotation_b": img_b["final_camera_rotation"]
                        })
                        yaw_range_counts[yaw_range] += 1
                        break
                        
                # 所有角度范围都已达到5对，可以提前退出
                if all(count >= 5 for count in yaw_range_counts.values()):
                    break
        
        # 对每个camera_uuid下的room最多取25对
        pairs.extend(valid_pairs[:25])

# 转换为 pandas DataFrame 并存储
pairs_df = pd.DataFrame(pairs)
pairs_csv_path = os.path.join(pickup_path, csv_name)
pairs_df.to_csv(pairs_csv_path, index=False)

# 复制选中的图片对到 pickUp 目录
# for pair in tqdm(pairs, desc="Copying images"):
#     img_a = pair["frame_num_a"]
#     img_b = pair["frame_num_b"]
#     room = pair["room"]
#     camera_uuid = pair["camera_uuid"]  # 获取camera_uuid
    
#     file_a = df[(df["frame_num"] == img_a) & (df["room"] == room) & (df["camera_uuid"] == camera_uuid)]["image_file"].values[0]
#     file_b = df[(df["frame_num"] == img_b) & (df["room"] == room) & (df["camera_uuid"] == camera_uuid)]["image_file"].values[0]
#     # new_name_a = f"{room}_frame_{img_a}.png"
#     # new_name_b = f"{room}_frame_{img_b}.png"
    
#     # shutil.copy(os.path.join(rgb_path, file_a), os.path.join(pickup_path, new_name_a))
#     # shutil.copy(os.path.join(rgb_path, file_b), os.path.join(pickup_path, new_name_b))
#     shutil.copy(os.path.join(rgb_path, file_a), os.path.join(pickup_path, file_a))
#     shutil.copy(os.path.join(rgb_path, file_b), os.path.join(pickup_path, file_b))
