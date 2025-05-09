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
            b2a_direction = row.get("b2a_direction", None)
            if b2a_direction is None or b2a_direction == "":
                b2a_direction = row.get("label", "")
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

def display_image_pairs_grid(csv_path, imgs_dir, N, save_path):
    """
    读取csv中前N对样本，将每对样本以A在上B在下的方式排列，所有样本横向排列，保存为一张大图。
    """
    import math
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
            b2a_direction = row.get("b2a_direction", None)
            if b2a_direction is None or b2a_direction == "":
                b2a_direction = row.get("label", "")
            difficulty = row["difficulty"]
            img_a_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_a}.png"
            img_b_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_b}.png"
            img_a_path = os.path.join(imgs_dir, img_a_name)
            img_b_path = os.path.join(imgs_dir, img_b_name)
            if not os.path.exists(img_a_path) or not os.path.exists(img_b_path):
                continue
            pairs.append((img_a_path, img_b_path, b2a_direction, difficulty))
            if len(pairs) >= N:
                break
    if not pairs:
        print("No valid image pairs found.")
        return
    # 读取所有图片
    imgs_a = [Image.open(a).convert("RGB") for a,_,_,_ in pairs]
    imgs_b = [Image.open(b).convert("RGB") for _,b,_,_ in pairs]
    # 统一图片尺寸
    w, h = imgs_a[0].size
    imgs_a = [img.resize((w, h)) for img in imgs_a]
    imgs_b = [img.resize((w, h)) for img in imgs_b]
    # 创建画布
    fig, axes = plt.subplots(2, N, figsize=(N*3, 6))
    for i in range(N):
        axes[0, i].imshow(imgs_a[i])
        axes[0, i].set_title(f"A\n{pairs[i][2]}\n{pairs[i][3]}", fontsize=10)
        axes[0, i].axis("off")
        axes[1, i].imshow(imgs_b[i])
        axes[1, i].set_title("B", fontsize=10)
        axes[1, i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    # csv_path = "ImagePairsFromPano/5classes_dataset/diverse_sample_csv/samples_1.csv"
    imgs_dir = "ImagePairsFromPano/5classes_dataset/rgb"
    # display_image_pairs(csv_path, imgs_dir)
    # 示例：保存前8对样本为一张大图
    save_path = "ImagePairsFromPano/5classes_dataset/diverse_sample_csv/image_pairs_1.png"
    # display_image_pairs_grid(csv_path, imgs_dir, N=10, save_path=save_path)
    for i in range(4):
        csv_path = f"ImagePairsFromPano/5classes_dataset/diverse_sample_csv/samples_{i}.csv"
        save_path = f"ImagePairsFromPano/5classes_dataset/diverse_sample_csv/image_pairs_{i}.png"
        display_image_pairs_grid(csv_path, imgs_dir, N=10, save_path=save_path)