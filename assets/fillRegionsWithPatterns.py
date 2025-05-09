import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as NetX
import os
from numba import jit, prange, uint8, int32, boolean, njit

def calculate_pattern_size(image_shape, scale_factor=0.05):
    """
    根据图像尺寸计算合适的图案大小
    
    :param image_shape: 图像的形状 (height, width)
    :param scale_factor: 图案大小相对于图像最小边的比例
    :return: 建议的图案大小
    """
    min_dimension = min(image_shape[0], image_shape[1])
    pattern_size = max(20, int(min_dimension * scale_factor))
    # 确保图案尺寸是偶数，便于创建对称图案
    if pattern_size % 2 != 0:
        pattern_size += 1
    return pattern_size

@jit(nopython=True)
def create_pattern(pattern_type, size=60):
    """
    创建黑白重复图案，使用numba加速
    
    :param pattern_type: 图案类型 (0: 水平条纹, 1: 垂直条纹, 2: 左斜线, 3: 右斜线, 4: 网格, 5: 点状)
    :param size: 图案的基本大小
    :return: 黑白图案图像 (size x size)
    """
    pattern = np.ones((size, size), dtype=np.uint8) * 255
    
    # 根据图案尺寸调整纹理密度
    stripe_width = max(2, size // 10)  # 条纹宽度，最小2像素
    grid_spacing = max(4, size // 5)    # 网格间距，最小4像素
    dot_spacing = max(3, size // 7)     # 点间距，最小3像素
    
    if pattern_type == 0:  # 水平条纹
        for i in range(0, size, stripe_width * 2):
            for j in range(size):
                if i < size and i + stripe_width < size:
                    pattern[i:i+stripe_width, j] = 0
    elif pattern_type == 1:  # 垂直条纹
        for j in range(0, size, stripe_width * 2):
            for i in range(size):
                if j < size and j + stripe_width < size:
                    pattern[i, j:j+stripe_width] = 0
    elif pattern_type == 2:  # 左斜线 (\)
        thickness = max(1, size // 25)  # 斜线粗细
        for offset in range(0, size * 2, stripe_width * 2):
            for i in range(size):
                for j in range(size):
                    if abs((i + j) - offset) < thickness:
                        pattern[i, j] = 0
    elif pattern_type == 3:  # 右斜线 (/)
        thickness = max(1, size // 25)  # 斜线粗细
        for offset in range(0, size * 2, stripe_width * 2):
            for i in range(size):
                for j in range(size):
                    if abs((i - j + size) - offset) < thickness:
                        pattern[i, j] = 0
    elif pattern_type == 4:  # 网格
        line_thickness = max(1, size//50)
        for i in range(0, size, grid_spacing):
            if i < size:
                end_i = min(i+line_thickness, size)
                for i_pos in range(i, end_i):
                    for j in range(size):
                        pattern[i_pos, j] = 0
        
        for j in range(0, size, grid_spacing):
            if j < size:
                end_j = min(j+line_thickness, size)
                for j_pos in range(j, end_j):
                    for i in range(size):
                        pattern[i, j_pos] = 0
    elif pattern_type == 5:  # 点状
        dot_size = max(1, size // 20)  # 点的大小
        for i in range(dot_size, size, dot_spacing * 2):
            for j in range(dot_size, size, dot_spacing * 2):
                for di in range(-dot_size, dot_size+1):
                    for dj in range(-dot_size, dot_size+1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            pattern[ni, nj] = 0
    
    return pattern

@njit(parallel=True)
def create_pattern_parallel(pattern_type, size=60):
    """
    并行创建黑白重复图案，使用numba加速
    
    :param pattern_type: 图案类型 (0: 水平条纹, 1: 垂直条纹, 2: 网格, 3: 点状)
    :param size: 图案的基本大小
    :return: 黑白图案图像 (size x size)
    """
    pattern = np.ones((size, size), dtype=np.uint8) * 255
    
    # 根据图案尺寸调整纹理密度
    stripe_width = max(2, size // 10)  # 条纹宽度，最小2像素
    grid_spacing = max(4, size // 5)    # 网格间距，最小4像素
    dot_spacing = max(3, size // 7)     # 点间距，最小3像素
    
    if pattern_type == 0:  # 水平条纹 - 可以并行化
        for i in prange(0, size, stripe_width * 2):
            if i < size and i + stripe_width < size:
                for j in prange(size):
                    pattern[i:i+stripe_width, j] = 0
    elif pattern_type == 1:  # 垂直条纹 - 可以并行化
        for j in prange(0, size, stripe_width * 2):
            if j < size and j + stripe_width < size:
                for i in prange(size):
                    pattern[i, j:j+stripe_width] = 0
    elif pattern_type == 4:  # 网格 - 部分可以并行化
        # 水平线
        for i in prange(0, size, grid_spacing):
            if i < size:
                end_i = min(i+stripe_width, size)
                for i_pos in range(i, end_i):
                    for j in range(size):
                        pattern[i_pos, j] = 0
        
        # 垂直线
        for j in prange(0, size, grid_spacing):
            if j < size:
                end_j = min(j+stripe_width, size)
                for j_pos in range(j, end_j):
                    for i in range(size):
                        pattern[i, j_pos] = 0
    elif pattern_type == 5:  # 点状 - 可以并行化处理点的位置
        dot_size = max(1, size // 20)  # 点的大小
        for i in prange(dot_size, size, dot_spacing * 2):
            for j in range(dot_size, size, dot_spacing * 2):
                for di in range(-dot_size, dot_size+1):
                    for dj in range(-dot_size, dot_size+1):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size:
                            pattern[ni, nj] = 0
    else:  # 其他图案仍使用原始的非并行版本
        return create_pattern(pattern_type, size)
    
    return pattern

@jit(nopython=True)
def build_adjacency_matrix(segmentation):
    """
    构建区域邻接矩阵，使用numba加速
    
    :param segmentation: 语义分割图
    :return: 邻接矩阵和唯一标签列表
    """
    # 获取所有唯一标签
    unique_labels = np.unique(segmentation)
    n_labels = len(unique_labels)
    
    # 创建邻接矩阵 (使用字典难以在numba中实现，改用邻接矩阵)
    adjacency = np.zeros((n_labels, n_labels), dtype=np.bool_)
    
    # 为了快速查找标签的索引，创建映射字典
    label_to_idx = np.zeros(256, dtype=np.int32) - 1  # 假设标签值小于256
    for i, label in enumerate(unique_labels):
        if label < 256:
            label_to_idx[label] = i
    
    # 计算邻接关系
    height, width = segmentation.shape
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
    
    for y in range(height):
        for x in range(width):
            current_label = segmentation[y, x]
            current_idx = label_to_idx[current_label] if current_label < 256 else -1
            
            if current_idx == -1:
                continue
                
            # 检查四个方向的邻居
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    neighbor_label = segmentation[ny, nx]
                    neighbor_idx = label_to_idx[neighbor_label] if neighbor_label < 256 else -1
                    
                    if neighbor_idx == -1 or current_idx == neighbor_idx:
                        continue
                        
                    adjacency[current_idx, neighbor_idx] = True
    
    return adjacency, unique_labels

def build_adjacency_graph(segmentation):
    """
    构建区域邻接图
    
    :param segmentation: 语义分割图
    :return: 区域邻接关系图
    """
    # 使用numba加速的函数构建邻接矩阵
    adjacency_matrix, unique_labels = build_adjacency_matrix(segmentation)
    
    # 创建图
    G = NetX.Graph()
    
    # 添加节点
    for label in unique_labels:
        G.add_node(label)
    
    # 根据邻接矩阵添加边
    n_labels = len(unique_labels)
    for i in range(n_labels):
        for j in range(i+1, n_labels):
            if adjacency_matrix[i, j] or adjacency_matrix[j, i]:
                G.add_edge(unique_labels[i], unique_labels[j])
    
    return G

def color_graph(graph):
    """
    对图进行着色（贪心算法）
    
    :param graph: 区域邻接图
    :return: 每个节点的颜色映射
    """
    colors = {}
    available_colors = list(range(6))  # 6种黑白图案
    
    # 对所有节点进行着色
    for node in graph:
        # 获取邻居节点已使用的颜色
        used_colors = set(colors.get(neighbor, -1) for neighbor in graph[node])
        
        # 从可用颜色中选择第一个未被邻居使用的颜色
        available = [color for color in available_colors if color not in used_colors]
        if available:
            colors[node] = available[0]
        else:
            # 如果需要更多颜色，可以增加pattern_type的范围
            colors[node] = len(available_colors)
            available_colors.append(len(available_colors))
    
    return colors

@jit(nopython=True)
def fill_pattern_region(result, pattern, mask, y_start, x_start, pattern_height, pattern_width):
    """
    用图案填充特定区域，使用numba加速
    
    :param result: 结果图像
    :param pattern: 填充图案
    :param mask: 区域掩码
    :param y_start, x_start: 起始坐标
    :param pattern_height, pattern_width: 图案尺寸
    """
    h, w = mask.shape
    
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:
                pattern_y = y % pattern_height
                pattern_x = x % pattern_width
                result[y_start + y, x_start + x] = pattern[pattern_y, pattern_x]

@njit(parallel=True)
def fill_pattern_region_parallel(result, pattern, mask, y_start, x_start, pattern_height, pattern_width):
    """
    并行用图案填充特定区域，使用numba加速
    
    :param result: 结果图像
    :param pattern: 填充图案
    :param mask: 区域掩码
    :param y_start, x_start: 起始坐标
    :param pattern_height, pattern_width: 图案尺寸
    """
    h, w = mask.shape
    
    for y in prange(h):
        for x in range(w):
            if mask[y, x] > 0:
                pattern_y = y % pattern_height
                pattern_x = x % pattern_width
                result[y_start + y, x_start + x] = pattern[pattern_y, pattern_x]

def fill_regions_with_patterns(segmentation_path, output_path, pattern_size=None, scale_factor=0.05):
    """
    使用黑白重复图案填充语义分割区域
    
    :param segmentation_path: 输入语义分割图路径
    :param output_path: 输出结果图像路径
    :param pattern_size: 图案的基本大小，如果为None则自动计算
    :param scale_factor: 当pattern_size为None时，图案大小相对于图像最小边的比例
    """
    # 读取语义分割图
    segmentation = cv2.imread(segmentation_path, cv2.IMREAD_GRAYSCALE)
    if segmentation is None:
        raise FileNotFoundError(f"无法加载图像：{segmentation_path}")
    
    # 如果未指定图案大小，则自动计算
    if pattern_size is None:
        pattern_size = calculate_pattern_size(segmentation.shape, scale_factor)
        print(f"根据图像尺寸 {segmentation.shape} 自动计算的图案大小: {pattern_size}x{pattern_size}")
    
    
    # 构建区域邻接图
    adjacency_graph = build_adjacency_graph(segmentation)
    
    # 对图进行着色
    color_mapping = color_graph(adjacency_graph)
    
    # 创建用于存储结果的图像
    height, width = segmentation.shape
    result = np.ones((height, width), dtype=np.uint8) * 255
    
    # 创建所有需要的图案 - 使用并行版本
    patterns = {}
    for color in set(color_mapping.values()):
        try:
            patterns[color] = create_pattern_parallel(color % 6, pattern_size)
        except:
            # 如果并行版本失败，回退到非并行版本
            patterns[color] = create_pattern(color % 6, pattern_size)
    
    # 用图案填充每个区域
    for label, color in color_mapping.items():
        pattern = patterns[color]
        pattern_height, pattern_width = pattern.shape
        
        # 创建区域掩码
        mask = (segmentation == label).astype(np.uint8)
        
        # 找到掩码的非零区域以减少处理范围
        non_zero_points = np.argwhere(mask > 0)
        if len(non_zero_points) == 0:
            continue
            
        y_min, x_min = non_zero_points.min(axis=0)
        y_max, x_max = non_zero_points.max(axis=0)
        
        # 将大掩码分割成小块进行处理
        block_size = pattern_height * 4  # 使用更大的块以减少函数调用开销
        for y in range(y_min, y_max + 1, block_size):
            for x in range(x_min, x_max + 1, block_size):
                # 计算当前区块的有效范围
                h = min(block_size, height - y)
                w = min(block_size, width - x)
                
                # 只在掩码区域内填充图案
                region_mask = mask[y:y+h, x:x+w]
                if np.any(region_mask):
                    try:
                        # 尝试使用并行版本
                        fill_pattern_region_parallel(result, pattern, region_mask, y, x, pattern_height, pattern_width)
                    except:
                        # 如果并行版本失败，回退到非并行版本
                        fill_pattern_region(result, pattern, region_mask, y, x, pattern_height, pattern_width)
    
    # 保存结果图像
    cv2.imwrite(output_path, result)
    print(f"填充图案结果已保存到：{output_path}")

def process_folders(folder_paths, scale_factor=0.05, pattern_size=None):
    """
    处理多个文件夹中的PNG图像，使用黑白重复图案填充语义分割区域
    
    :param folder_paths: 包含PNG图像的文件夹路径列表
    :param scale_factor: 图案大小相对于图像最小边的比例
    :param pattern_size: 图案的基本大小，如果为None则自动计算
    """
    import time
    total_files = 0
    processed_files = 0
    start_time = time.time()
    
    # 计算总文件数
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
            total_files += len(png_files)
    
    print(f"开始处理，总共发现 {total_files} 个PNG文件")
    
    for folder_path in folder_paths:
        if not os.path.exists(folder_path):
            print(f"警告: 文件夹不存在 - {folder_path}")
            continue
            
        # 创建输出文件夹（与输入文件夹同级，名称为 Texture）
        output_folder = os.path.join(os.path.dirname(folder_path), "Texture")
        
        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取文件夹中的所有PNG文件
        png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
        
        if not png_files:
            print(f"警告: 在{folder_path}中未找到PNG文件")
            continue
            
        print(f"正在处理文件夹: {folder_path}, 发现 {len(png_files)} 个PNG文件")
        
        # 处理每个PNG文件
        for i, png_file in enumerate(png_files):
            input_path = os.path.join(folder_path, png_file)
            
            # 修改文件名，将最后一个_后的标签改为texture
            filename_parts = os.path.splitext(png_file)[0].split('_')
            if len(filename_parts) > 1:
                # 将最后一个部分替换为"texture"
                filename_parts[-1] = "texture"
                output_filename = "_".join(filename_parts) + ".png"
            else:
                # 如果文件名中没有下划线，则直接添加_texture后缀
                output_filename = filename_parts[0] + "_texture.png"
            
            output_path = os.path.join(output_folder, output_filename)
            
            # 如果文件已存在，跳过处理
            if os.path.exists(output_path):
                print(f"文件 {output_filename} 已存在，跳过处理")
                processed_files += 1
                continue
                
            try:
                file_start_time = time.time()
                print(f"[{processed_files + 1}/{total_files}] 处理文件: {png_file}...")
                fill_regions_with_patterns(input_path, output_path, pattern_size, scale_factor)
                processed_files += 1
                
                # 计算并显示进度
                elapsed = time.time() - start_time
                file_elapsed = time.time() - file_start_time
                avg_time_per_file = elapsed / processed_files
                remaining_files = total_files - processed_files
                estimated_remaining = remaining_files * avg_time_per_file
                
                print(f"文件处理完成，耗时: {file_elapsed:.2f}秒")
                print(f"总进度: {processed_files}/{total_files} ({processed_files/total_files*100:.1f}%), "
                      f"预计剩余时间: {estimated_remaining/60:.1f}分钟")
                
            except Exception as e:
                print(f"处理文件 {png_file} 时出错: {str(e)}")
    
    # 显示总计时
    total_time = time.time() - start_time
    print(f"所有文件处理完成，总耗时: {total_time/60:.2f}分钟")

if __name__ == "__main__":
    # 要处理的文件夹路径列表
    folders_to_process = [
        "area_1/pano/semantic",
        "area_2/pano/semantic",
        "area_3/pano/semantic",
        "area_4/pano/semantic",
        "area_5a/pano/semantic",
        "area_5b/pano/semantic",
        "area_6/pano/semantic",
        # 添加更多文件夹路径...
    ]
    
    # 使用默认参数处理所有文件夹
    process_folders(folders_to_process)
    
    # 或者指定自定义参数
    # process_folders(folders_to_process, scale_factor=0.08, pattern_size=100)