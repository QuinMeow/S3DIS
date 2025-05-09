import os
import cv2
import py360convert
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import glob

def create_directories(output_dir):
    """创建输出目录"""
    os.makedirs(output_dir, exist_ok=True)

def load_pairs_from_csv(csv_path):
    """从CSV文件加载图像对信息"""
    return pd.read_csv(csv_path)

def build_panorama_map(pano_dirs, pano_type):
    """
    扫描所有提供的全景图目录，构建camera_uuid和room到全景图路径的映射
    
    Args:
        pano_dirs: 全景图目录列表
        pano_type: 全景图类型，如"depth"、"normal"等
        
    Returns:
        字典，键为(camera_uuid, room)，值为对应全景图的完整路径
    """
    panorama_map = {}
    print(f"Scanning panorama directories for {pano_type} images...")
    
    for pano_dir in pano_dirs:
        pano_dir = os.path.join(pano_dir, pano_type)
        # 确保这是一个有效的目录
        if not os.path.isdir(pano_dir):
            print(f"Warning: {pano_dir} is not a valid directory, skipping...")
            continue
        
        # 搜索目录中的全景图
        search_pattern = os.path.join(pano_dir, f"*_{pano_type}.png")
        pano_files = glob.glob(search_pattern)
        
        for pano_path in pano_files:
            pano_filename = os.path.basename(pano_path)
            parts = pano_filename.split("_")
            
            # 检查文件名格式是否符合预期
            if len(parts) >= 5 and parts[0] == "camera":
                camera_uuid = parts[1]
                room = parts[2] + "_" + parts[3]
                
                # 将全景图添加到映射中
                key = (camera_uuid, room)
                panorama_map[key] = pano_path
                
    print(f"Found {len(panorama_map)} panorama images of type {pano_type}.")
    return panorama_map

def process_panorama_images(pairs_df, panorama_map, output_dir, pano_type, image_size=[512, 512]):
    """从全景图中截取特定视角"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录已处理的全景图，避免重复读取
    processed_panos = {}
    
    # 记录成功处理的图像对数量
    success_count = 0
    skipped_count = 0
    
    # 跟踪已经生成的图片，避免重复生成
    processed_frames = set()
    
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"Processing {pano_type} images"):
        room = row['room'][:-2]  # 去掉最后两个字符
        camera_uuid = row['camera_uuid']
        
        # 构建键，查找对应的全景图路径
        key = (camera_uuid, room)
        
        if key not in panorama_map:
            print(f"Warning: No {pano_type} panorama found for camera {camera_uuid} in room {room}, skipping...")
            skipped_count += 1
            continue
        
        pano_path = panorama_map[key]
        
        # 如果此全景图尚未处理过，则读取它
        if pano_path not in processed_panos:
            pano_img = cv2.imread(pano_path)
            if pano_img is None:
                print(f"Error: Could not read {pano_path}, skipping...")
                skipped_count += 1
                continue
            processed_panos[pano_path] = pano_img
        else:
            pano_img = processed_panos[pano_path]
        
        # 处理A和B两个视角
        frames_processed = 0
        for frame_type in ['a', 'b']:
            # 获取对应的参数
            yaw = row[f'yaw_{frame_type}']
            pitch = row[f'pitch_{frame_type}']
            frame_num = int(row[f'frame_num_{frame_type}'])
            fov = float(row['fov'])
            
            # 构建图像标识符
            frame_id = f"{camera_uuid}_{room}_frame_{frame_num}"
            filename = f"perspective_{frame_id}.png"
            save_path = os.path.join(output_dir, filename)
            
            # 检查该图片是否已经处理过
            if frame_id in processed_frames:
                frames_processed += 1
                continue
                
            # 生成投影图像
            persp_img = py360convert.e2p(pano_img, fov, -yaw, pitch, image_size)
            
            # 保存图像
            cv2.imwrite(save_path, persp_img)
            
            # 将图片标记为已处理
            processed_frames.add(frame_id)
            frames_processed += 1
            
        # 只有当两个帧都成功处理时才增加成功计数
        if frames_processed > 0:
            success_count += 1
            
    total_frames = len(processed_frames)
    print(f"Successfully processed {success_count} image pairs from {pano_type} panoramas")
    print(f"Generated {total_frames} unique perspective images")
    print(f"Skipped {skipped_count} image pairs due to missing panoramas or errors")
    return success_count, skipped_count, total_frames

def main():

    csv_path = "ImagePairsFromPano/5classes_dataset/test.csv"
    pano_dirs = [
        "area_1/pano",
        "area_2/pano",
        "area_3/pano",
        "area_4/pano",
        "area_5a/pano",
        "area_5b/pano",
        "area_6/pano",
    ]
    pano_type = "texture"  
    output_dir = "ImagePairsFromPano/5classes_dataset"
    image_size = [512, 512]  # 默认图像大小

    # 加载CSV数据
    pairs_df = load_pairs_from_csv(csv_path)
    print(f"Loaded {len(pairs_df)} image pairs from {csv_path}")
    
    # 构建全景图映射
    panorama_map = build_panorama_map(pano_dirs, pano_type)
    
    # 确定输出目录
    output_dir = os.path.join(output_dir, pano_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理全景图
    process_panorama_images(pairs_df, panorama_map, output_dir, pano_type, image_size)

if __name__ == "__main__":
    main()