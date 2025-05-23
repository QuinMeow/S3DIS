import os
import csv
import shutil
import argparse

def copy_image_pairs_from_csv(csv_path, imgs_dir, out_dir, N=None):
    """
    从csv读取图片对，将图片复制到out_dir。
    :param csv_path: csv文件路径
    :param imgs_dir: 原始图片文件夹
    :param out_dir: 目标文件夹
    :param N: 最多复制N对（2N张）图片，None表示全部
    """
    os.makedirs(out_dir, exist_ok=True)
    pairs = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["camera_uuid"].lower() == "camera_uuid":
                continue
            camera_uuid = row["camera_uuid"]
            room = row["room"]
            frame_num_a = row["frame_num_a"]
            frame_num_b = row["frame_num_b"]
            img_a_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_a}.png"
            img_b_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_b}.png"
            img_a_path = os.path.join(imgs_dir, img_a_name)
            img_b_path = os.path.join(imgs_dir, img_b_name)
            if not os.path.exists(img_a_path) or not os.path.exists(img_b_path):
                print(f"Images not found: {img_a_path}, {img_b_path}")
                continue
            pairs.append((img_a_path, img_b_path, img_a_name, img_b_name))
            if N is not None and len(pairs) >= N:
                break
    if not pairs:
        print("No valid image pairs found.")
        return
    # 复制图片
    for img_a_path, img_b_path, img_a_name, img_b_name in pairs:
        shutil.copy(img_a_path, os.path.join(out_dir, img_a_name))
        shutil.copy(img_b_path, os.path.join(out_dir, img_b_name))
    print(f"Copied {len(pairs)*2} images to {out_dir}")

if __name__ == "__main__":
    # 直接硬编码路径，无需命令行参数
    imgs_dir = "ImagePairsFromPano/5classes_dataset/rgb"
    N = None  # 可根据需要修改
    for i in range(4):
        csv_path = f"ImagePairsFromPano/5classes_dataset/diverse_sample_csv/samples_{i}.csv"
        out_dir = f"ImagePairsFromPano/5classes_dataset/diverse_sample_csv/copied_imgs_{i}"
        copy_image_pairs_from_csv(csv_path, imgs_dir, out_dir, N)
