import os
import cv2
import py360convert
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from math import pi

def generate_perspective_images(pano_image, pano_data, fov, pitch, yaw, img_size=[1080, 1080]):
    persp_info = []
    for theta in yaw:
        persp_img = py360convert.e2p(pano_image, fov, theta, pitch, img_size)
        filename = f"perspective_{theta}.jpg"
        persp_info.append((theta, filename))
        # 保存透视图像
        persp_img.save(filename)
    return persp_info

def process_pose_data(pose_path):
    """处理姿态数据文件"""
    pose_data = {}
    for pose_file in tqdm(os.listdir(pose_path), desc="Processing pose files"):
        if pose_file.endswith(".json"):
            pose_filepath = os.path.join(pose_path, pose_file)
            with open(pose_filepath, "r") as f:
                data = json.load(f)
                pose_data[pose_file] = data
    return pose_data

def process_image_data(rgb_path, pose_data):
    """处理图像文件并匹配姿态数据"""
    images_data = []
    for img_file in tqdm(os.listdir(rgb_path), desc="Processing image files"):
        if img_file.endswith(".png"):
            parts = img_file.split("_")
            camera_uuid = parts[1]
            room = parts[2] + "_" + parts[3]
            pose_file = f"camera_{camera_uuid}_{room}_frame_equirectangular_domain_pose.json"
            if pose_file in pose_data:
                data = pose_data[pose_file]
                images_data.append({
                    "image_file": img_file,
                    "room": data["room"],
                    "camera_uuid": camera_uuid,
                    "camera_location": data["camera_location"],
                    "final_camera_rotation": data["final_camera_rotation"],
                    "fx": data["camera_k_matrix"][0][0],
                    "fy": data["camera_k_matrix"][1][1]
                })
    return images_data

def generate_image_pairs(images_data, rgb_path, output_rgb_path, task_types, num_base_images, fov, image_size):
    """生成图像对"""
    pairs = []
    
    for img_data in tqdm(images_data, desc="Creating perspective image pairs"):
        frame_id = 0
        img_path = os.path.join(rgb_path, img_data["image_file"])
        img = cv2.imread(img_path)
            
        for i in range(num_base_images):
            # 生成基准图片a的参数，pitch固定为0，yaw随机
            pitch_a = 0  # 固定pitch为0
            yaw_a = np.random.uniform(-180, 180)
            
            # 生成和保存基准图A
            persp_img_a = py360convert.e2p(img, fov, -yaw_a, pitch_a, image_size)
            filename_a = f"perspective_{img_data['camera_uuid']}_{img_data['room'][:-2]}_frame_{frame_id}.png"
            save_img_path_a = os.path.join(output_rgb_path, filename_a)
            cv2.imwrite(save_img_path_a, persp_img_a)
            frame_id_a = frame_id
            frame_id += 1
            
            # 为每种任务类型生成一个对应的图B
            for b2a_direction, params in task_types.items():
                # 跳过超出数量的生成
                if i >= params["count"]:
                    continue
                
                # 根据任务类型设置pitch_diff和yaw_diff
                if b2a_direction == "Up":
                    yaw_diff = 0
                    pitch_diff = np.random.uniform(params["pitch_diff_range"][0], params["pitch_diff_range"][1])
                elif b2a_direction == "Down":
                    yaw_diff = 0
                    pitch_diff = np.random.uniform(params["pitch_diff_range"][0], params["pitch_diff_range"][1])
                elif b2a_direction == "Left":
                    yaw_diff = np.random.uniform(params["yaw_diff_range"][0], params["yaw_diff_range"][1])
                    pitch_diff = 0
                elif b2a_direction == "Right":
                    yaw_diff = np.random.uniform(params["yaw_diff_range"][0], params["yaw_diff_range"][1])
                    pitch_diff = 0
                else:  # Unknown  
                    yaw_diff = np.random.uniform(params["yaw_diff_range"][0], params["yaw_diff_range"][1])
                    if np.random.choice([True, False]):  # 随机选择正负
                        yaw_diff = -yaw_diff
                    pitch_diff = np.random.uniform(params["pitch_diff_range"][0], params["pitch_diff_range"][1])
                
                # 计算B图的pitch和yaw
                pitch_b = pitch_a + pitch_diff
                yaw_b = yaw_a + yaw_diff
                
                # 确保pitch在[-30, 30]范围内
                pitch_b = max(min(pitch_b, 30), -30)
                
                # 确定任务难度
                difficulty = "Unknown"
                if b2a_direction != "Unknown":
                    abs_yaw_diff = abs(yaw_diff)
                    abs_pitch_diff = abs(pitch_diff)
                    
                    if (abs_yaw_diff < 15 and abs_yaw_diff != 0 ) or (abs_pitch_diff < 10 and abs_pitch_diff != 0):
                        difficulty = "Easy"
                    elif (abs_yaw_diff < 45 and abs_yaw_diff != 0) or (abs_pitch_diff < 20 and abs_pitch_diff != 0):
                        difficulty = "Normal"
                    else:
                        difficulty = "Hard"
                
                # 生成和保存B图像
                persp_img_b = py360convert.e2p(img, fov, -yaw_b, pitch_b, image_size)
                filename_b = f"perspective_{img_data['camera_uuid']}_{img_data['room'][:-2]}_frame_{frame_id}.png"
                save_img_path_b = os.path.join(output_rgb_path, filename_b)
                cv2.imwrite(save_img_path_b, persp_img_b)
                frame_id_b = frame_id
                frame_id += 1
                
                # 添加图像对信息
                pairs.append({
                    "room": img_data["room"],
                    "camera_uuid": img_data["camera_uuid"],
                    "camera_location": img_data["camera_location"],
                    "frame_num_a": frame_id_a,
                    "frame_num_b": frame_id_b,
                    "yaw_a": yaw_a,
                    "pitch_a": pitch_a,
                    "yaw_b": yaw_b,
                    "pitch_b": pitch_b,
                    "yaw_diff": yaw_diff,
                    "pitch_diff": pitch_diff,
                    "b2a_direction": b2a_direction,
                    "difficulty": difficulty,
                    "fov": fov,
                    "final_camera_rotation_a": img_data["final_camera_rotation"] + [pitch_a/180*pi, 0, yaw_a/180*pi],
                    "final_camera_rotation_b": img_data["final_camera_rotation"] + [pitch_b/180*pi, 0, yaw_b/180*pi]
                })
    
    return pairs

def process_dataset(dataset_name, base_path, output_base_dir, task_types, num_base_images, fov=65, image_size=[512, 512]):
    """处理单个数据集"""
    print(f"Processing dataset: {dataset_name}")
    
    # 设置路径
    data_path = os.path.join(base_path, dataset_name, "pano")
    rgb_path = os.path.join(data_path, "rgb")
    pose_path = os.path.join(data_path, "pose")
    
    # 创建输出目录
    output_dir = os.path.join(output_base_dir, "5classes_dataset")
    os.makedirs(output_dir, exist_ok=True)
    output_rgb_path = os.path.join(output_dir, "rgb")
    os.makedirs(output_rgb_path, exist_ok=True)
    
    # 处理数据
    pose_data = process_pose_data(pose_path)
    images_data = process_image_data(rgb_path, pose_data)
    pairs = generate_image_pairs(images_data, rgb_path, output_rgb_path, task_types, num_base_images, fov, image_size)
    
    # 保存结果
    csv_name = f"{dataset_name}_pairs.csv"
    pairs_df = pd.DataFrame(pairs)
    pairs_csv_path = os.path.join(output_dir, csv_name)
    pairs_df.to_csv(pairs_csv_path, index=False)
    
    print(f"Processed {len(pairs)} image pairs for {dataset_name}")
    return pairs_df


if __name__ == "__main__":
    # 基本配置
    base_path = "/data2/zyz/S3DIS"
    output_base_dir = "/data2/zyz/S3DIS/ImagePairsFromPano"

    # 视角参数设置
    fov = 65    # 视场角（度）
    image_size = [512, 512]
    num_base_images = 10  # 基准图的数量

    # 定义任务类型和每种类型的样本数量
    task_types = {
        "Up": {"count": 10, "yaw_diff": 0, "pitch_diff_range": (0, 30)},
        "Down": {"count": 10, "yaw_diff": 0, "pitch_diff_range": (-30, 0)},
        "Left": {"count": 10, "yaw_diff_range": (0, 60), "pitch_diff": 0},
        "Right": {"count": 10, "yaw_diff_range": (-60, 0), "pitch_diff": 0},
        "Unknown": {"count": 5, "yaw_diff_range": (90, 180), "pitch_diff_range": (-30, 30)}
    }

    # 指定要处理的数据集
    dataset_names = ["area_1", "area_2", "area_3", "area_4", "area_5a", "area_5b" ,"area_6"]

    # 处理每个数据集
    for dataset_name in dataset_names:
        process_dataset(dataset_name, base_path, output_base_dir, task_types, num_base_images, fov, image_size)
    print("All datasets processed successfully!")







