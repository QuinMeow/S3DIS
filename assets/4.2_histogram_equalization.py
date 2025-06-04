'''
对深度图做直方图均衡化
'''
import cv2
import numpy as np
import sys
import os
import csv
from PIL import Image
from tqdm import tqdm

def to_8bit(img):
    if img.dtype == np.uint16:
        # 只用1%~99%分位数做归一化，减少极端值影响
        vmin = np.percentile(img, 10)
        vmax = np.percentile(img, 95)
        img = np.clip(img, vmin, vmax)
        img_8bit = cv2.convertScaleAbs(img, alpha=255.0/(vmax-vmin), beta=-255.0*vmin/(vmax-vmin))
        return img_8bit
    elif img.dtype == np.float32 or img.dtype == np.float64:
        img_8bit = cv2.convertScaleAbs(img, alpha=255.0)
        return img_8bit
    elif img.dtype == np.uint8:
        return img
    else:
        raise ValueError(f"不支持的图像数据类型: {img.dtype}")

def histogram_equalization(input_path, output_path=None):
    # 读取图片
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"无法读取图片: {input_path}")
        return

    # 判断图片通道数
    if len(img.shape) == 2:
        # 灰度图
        img_8bit = to_8bit(img)
        eq_img = cv2.equalizeHist(img_8bit)
    elif len(img.shape) == 3:
        # 彩色图，对每个通道分别均衡化
        channels = cv2.split(img)
        eq_channels = [cv2.equalizeHist(to_8bit(ch)) for ch in channels]
        eq_img = cv2.merge(eq_channels)
    else:
        print("不支持的图片格式")
        return

    # 输出路径
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + '_equalized' + ext

    cv2.imwrite(output_path, eq_img)
    print(f"均衡化后的图片已保存到: {output_path}")

def get_equalization_lut(img_8bit):
    # 计算直方图
    hist = cv2.calcHist([img_8bit], [0], None, [256], [0,256]).flatten()
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    lut = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
    lut = np.ma.filled(lut, 0).astype('uint8')
    return lut

def equalize_pairs_from_csv(csv_path, imgs_dir, save_dir):
    """
    对于每一张A，先均衡化A，保存A的LUT，所有与A配对的B都用A的LUT做均衡化。
    """
    os.makedirs(save_dir, exist_ok=True)
    # 先统计总行数用于tqdm
    with open(csv_path, newline='') as f:
        total = sum(1 for _ in f) - 1  # 减去表头
    # 先按A分组
    pairs_by_A = {}
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
            pairs_by_A.setdefault(img_a_name, []).append(img_b_name)
    # LUT缓存
    lut_cache = {}
    for img_a_name, b_list in tqdm(pairs_by_A.items(), desc='Processing A images'):
        img_a_path = os.path.join(imgs_dir, img_a_name)
        if not os.path.exists(img_a_path):
            print(f"Image A not found: {img_a_path}")
            continue
        img_a = cv2.imread(img_a_path, cv2.IMREAD_UNCHANGED)
        if img_a is None:
            print(f"Failed to read: {img_a_path}")
            continue
        # 灰度或单通道
        if len(img_a.shape) == 2:
            img_a_8bit = to_8bit(img_a)
            lut = get_equalization_lut(img_a_8bit)
            eq_a = cv2.LUT(img_a_8bit, lut)
            cv2.imwrite(os.path.join(save_dir, img_a_name), eq_a)
            lut_cache[img_a_name] = lut
        elif len(img_a.shape) == 3:
            eq_a_channels = []
            lut_channels = []
            for ch in cv2.split(img_a):
                ch_8bit = to_8bit(ch)
                lut = get_equalization_lut(ch_8bit)
                eq_a_channels.append(cv2.LUT(ch_8bit, lut))
                lut_channels.append(lut)
            eq_a = cv2.merge(eq_a_channels)
            cv2.imwrite(os.path.join(save_dir, img_a_name), eq_a)
            lut_cache[img_a_name] = lut_channels
        else:
            print(f"不支持的图片格式: {img_a_path}")
            continue
        # 处理所有B
        for img_b_name in b_list:
            img_b_path = os.path.join(imgs_dir, img_b_name)
            if not os.path.exists(img_b_path):
                print(f"Image B not found: {img_b_path}")
                continue
            img_b = cv2.imread(img_b_path, cv2.IMREAD_UNCHANGED)
            if img_b is None:
                print(f"Failed to read: {img_b_path}")
                continue
            if len(img_a.shape) == 2:
                img_b_8bit = to_8bit(img_b)
                eq_b = cv2.LUT(img_b_8bit, lut_cache[img_a_name])
            elif len(img_a.shape) == 3:
                eq_b_channels = []
                for ch, lut in zip(cv2.split(img_b), lut_cache[img_a_name]):
                    ch_8bit = to_8bit(ch)
                    eq_b_channels.append(cv2.LUT(ch_8bit, lut))
                eq_b = cv2.merge(eq_b_channels)
            else:
                print(f"不支持的图片格式: {img_b_path}")
                continue
            cv2.imwrite(os.path.join(save_dir, img_b_name), eq_b)

if __name__ == "__main__":
    
    input_path = "/data2/zyz/S3DIS/area_2/pano/depth/camera_0a2acab6ce7b4cbdb431b640645eadfd_office_7_frame_equirectangular_domain_depth.png"
    output_path = "equaliztion.png"
    # histogram_equalization(input_path, output_path)
    # 示例用法（可在main中调用）
    csv_path = 'ImagePairsFromPano/5classes_dataset/test.csv'
    imgs_dir = 'ImagePairsFromPano/5classes_dataset/depth'
    save_dir = 'ImagePairsFromPano/5classes_dataset/depth_pretty'
    equalize_pairs_from_csv(csv_path, imgs_dir, save_dir)