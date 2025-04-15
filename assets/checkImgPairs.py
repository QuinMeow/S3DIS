import os
import csv
import argparse
from PIL import Image
import matplotlib.pyplot as plt

def display_image_pairs(csv_path, imgs_dir):
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["camera_uuid"].lower() == "camera_uuid":
                continue
            # ...existing code for读取csv数据...
            camera_uuid = row["camera_uuid"]
            room = row["room"]
            frame_num_a = row["frame_num_a"]
            frame_num_b = row["frame_num_b"]
            b2a_direction = row["b2a_direction"]
            difficulty = row["difficulty"]

            # 根据约定格式生成图片文件名
            img_a_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_a}.png"
            img_b_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_b}.png"
            img_a_path = os.path.join(imgs_dir, img_a_name)
            img_b_path = os.path.join(imgs_dir, img_b_name)

            if not os.path.exists(img_a_path) or not os.path.exists(img_b_path):
                print(f"Images not found: {img_a_path}, {img_b_path}")
                continue

            img_a = Image.open(img_a_path).convert("RGB")
            img_b = Image.open(img_b_path).convert("RGB")

            # 使用 matplotlib 并排展示图片，并在图像上附加相对位置关系信息
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_a)
            axes[0].set_title("Image A")
            axes[0].axis("off")
            axes[1].imshow(img_b)
            axes[1].set_title("Image B")
            axes[1].axis("off")
            plt.suptitle(f"Relative position: {b2a_direction}\nDifficulty: {difficulty}", fontsize=16)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    csv_path = "ImagePairsFromPano/5classes_dataset/test.csv"
    imgs_dir = "ImagePairsFromPano/5classes_dataset/semantic_pretty"
    display_image_pairs(csv_path, imgs_dir)
