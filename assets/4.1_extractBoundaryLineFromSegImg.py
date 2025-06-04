'''
从多个文件夹的语义分割图中提取边界线并保存为新图像
'''
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def extract_segmentation_boundaries(segmentation_path, output_path):
    """
    从语义分割图中提取边界线并保存为新图像。

    :param segmentation_path: 输入语义分割图的路径
    :param output_path: 输出边界线图的保存路径
    """
    # 读取语义分割图
    segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
    if segmentation is None:
        raise FileNotFoundError(f"无法加载图像：{segmentation_path}")

    # 创建一个空白图像用于存储边界
    boundaries = np.zeros_like(segmentation)

    # 获取所有唯一的类别标签
    unique_labels = np.unique(segmentation)

    # 遍历每个类别，提取边界
    for label in unique_labels:
        # 创建二值掩码
        mask = (segmentation == label).astype(np.uint8)

        # 使用Sobel算子检测边缘
        sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely).astype(np.uint8)

        # 将边界叠加到结果图像中
        boundaries = cv2.bitwise_or(boundaries, edges)

    # 将边界线图的颜色值改为255（白色）
    boundaries[boundaries > 0] = 255

    # 保存边界线图
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, boundaries)
    print(f"边界线图已保存到：{output_path}")

def process_folders(folder_paths):
    """
    处理多个文件夹中的PNG图像，提取分割边界线并保存到同级的新建文件夹中
    
    :param folder_paths: 包含PNG图像的文件夹路径列表
    """
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在 - {folder_path}")
            continue
            
        # 创建输出文件夹（与输入文件夹同级，名称为 原文件夹名_boundaries）
        # folder_name = os.path.basename(os.path.normpath(folder_path))
        output_folder = os.path.join(os.path.dirname(folder_path), "boundaries")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取文件夹中的所有PNG文件
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        
        if not png_files:
            print(f"警告: 在{folder_path}中未找到PNG文件")
            continue
            
        print(f"正在处理文件夹: {folder_path}, 发现 {len(png_files)} 个PNG文件")
        
        # 处理每个PNG文件
        for png_file in png_files:
            input_path = os.path.join(folder_path, png_file)
            
            # 修改文件名，将最后一个_后的标签改为boundary
            filename_parts = os.path.splitext(png_file)[0].split('_')
            if len(filename_parts) > 1:
                # 将最后一个部分替换为"boundary"
                filename_parts[-1] = "boundary"
                output_filename = "_".join(filename_parts) + ".png"
            else:
                # 如果文件名中没有下划线，则直接添加_boundary后缀
                output_filename = filename_parts[0] + "_boundary.png"
            
            output_path = os.path.join(output_folder, output_filename)
            
            try:
                extract_segmentation_boundaries(input_path, output_path)
            except Exception as e:
                print(f"处理文件 {png_file} 时出错: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 要处理的文件夹路径列表
    folders_to_process = [
        "area_1/pano/semantic",
        "area_2/pano/semantic",
        "area_4/pano/semantic",
        "area_5a/pano/semantic",
        "area_5b/pano/semantic",
        "area_6/pano/semantic",
        # 添加更多文件夹路径...
    ]
    
    process_folders(folders_to_process)