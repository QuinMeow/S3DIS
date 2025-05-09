import os
import csv
import argparse
import random
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import TextBox
from datetime import datetime

def read_image_pairs(csv_path):
    """从CSV文件中读取图像对数据"""
    image_pairs = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["camera_uuid"].lower() == "camera_uuid":
                continue
            
            image_pairs.append({
                'camera_uuid': row["camera_uuid"],
                'room': row["room"],
                'frame_num_a': row["frame_num_a"],
                'frame_num_b': row["frame_num_b"],
                'b2a_direction': row["b2a_direction"],
                'difficulty': row["difficulty"]
            })
    return image_pairs

def sample_image_pairs(image_pairs, num_samples):
    """随机抽取指定数量的图像对"""
    if num_samples > len(image_pairs):
        print(f"警告: 请求的样本数 {num_samples} 超过可用样本数 {len(image_pairs)}。使用所有可用样本。")
        return image_pairs
    return random.sample(image_pairs, num_samples)

def load_images(pair, imgs_dir):
    """根据图像对信息加载图像"""
    camera_uuid = pair["camera_uuid"]
    room = pair["room"]
    frame_num_a = pair["frame_num_a"]
    frame_num_b = pair["frame_num_b"]
    
    img_a_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_a}.png"
    img_b_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_b}.png"
    img_a_path = os.path.join(imgs_dir, img_a_name)
    img_b_path = os.path.join(imgs_dir, img_b_name)
    
    if not os.path.exists(img_a_path) or not os.path.exists(img_b_path):
        print(f"图像未找到: {img_a_path}, {img_b_path}")
        return None, None
    
    img_a = Image.open(img_a_path).convert("RGB")
    img_b = Image.open(img_b_path).convert("RGB")
    
    return img_a, img_b

def key_to_direction(key):
    """将按键转换为方向字符串"""
    key_map = {
        "left": "left",
        "right": "right",
        "up": "up",  
        "down": "down",
        " ": "unknown"  # 空格键表示unknown类别
    }
    return key_map.get(key, None)

def save_results(results, output_path, participant_name):
    """保存测试结果到CSV文件"""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if participant_name:
        filename = f"{participant_name}_{timestamp}.csv"
    else:
        filename = f"{timestamp}.csv"
        
    output_path = os.path.join(output_path, filename)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['camera_uuid', 'room', 'frame_num_a', 'frame_num_b', 
                     'b2a_direction', 'user_answer', 'is_correct', 'response_time', 'difficulty']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print(f"结果已保存至 {output_path}")

def run_test(csv_path, imgs_dir, num_samples=10, output_path=None):
    """运行空间关系测试程序"""
    image_pairs = read_image_pairs(csv_path)
    sampled_pairs = sample_image_pairs(image_pairs, num_samples)
    results = []
    font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14) # 中文字体
    plt.ion()  # 启用交互模式
    
    # 创建单一窗口用于整个测试过程
    fig = plt.figure(figsize=(12, 6))
    
    # 显示任务背景信息和说明
    plt.clf()
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # 任务说明文本
    welcome_text = """
    空间关系判断测试
    
    测试目的：
    评估人类对空间场景中物体相对位置关系的感知能力。
    
    测试流程：
    1. 每次将展示两张场景图片 A 和 B，请观察它们
    2. 图片显示2秒后将消失
    3. 请判断图片B相对于图片A的方位关系
    （等价于游戏中控制镜头从A位置看向B位置）
    4. 使用键盘方向键回答：
       · ← 左侧 (left)
       · → 右侧 (right)
       · ↑ 前方 (up)
       · ↓ 后方 (down)
       · 空格键 表示未知方向
    
    请在下方输入您的姓名（英文），按回车键开始测试...
    """
    
    ax.text(0.5, 0.65, welcome_text, fontsize=14, ha='center', 
            va='center', transform=ax.transAxes, fontproperties=font)
    
    # 添加姓名输入框
    plt.subplots_adjust(bottom=0.2)
    axbox = plt.axes([0.3, 0.15, 0.4, 0.05])  # 设置文本框位置
    text_box = TextBox(axbox, 'name: ', initial='')
    
    participant_name = ''
    start_test = False
    
    def submit_name(text):
        nonlocal participant_name, start_test
        participant_name = text
        start_test = True
    
    text_box.on_submit(submit_name)
    plt.draw()
    
    # 等待用户输入姓名并回车
    while not start_test:
        plt.pause(0.1)
    
    # 清理TextBox相关资源以避免错误
    # plt.disconnect(text_box.cid)  # 断开TextBox事件处理器
    if hasattr(text_box, 'observers'):
        text_box.disconnect_events()  # 对于较新版本的matplotlib
    
    # 完全清理界面
    plt.clf()
    plt.cla()
    
    # 开始测试
    for i, pair in enumerate(sampled_pairs):
        img_a, img_b = load_images(pair, imgs_dir)
        if img_a is None or img_b is None:
            continue
        
        # 清空当前图表并创建新的子图
        plt.clf()
        fig.suptitle("", fontsize=20)
        axes = fig.subplots(1, 2)
        plt.subplots_adjust(top=0.85)  # 留出空间放倒计时和提示
        
        # 3秒倒计时
        for count in range(3, 0, -1):
            fig.suptitle(f"准备... {count}", fontsize=20, color='blue', fontproperties=font)
            plt.draw()
            plt.pause(1)
        
        # 显示图像2秒
        axes[0].imshow(img_a)
        axes[0].set_title("图像 A", fontsize=16, fontproperties=font)
        axes[0].axis("off")
        axes[1].imshow(img_b)
        axes[1].set_title("图像 B", fontsize=16, fontproperties=font)
        axes[1].axis("off")
        fig.suptitle("请判断图片B相对于图片A的方位", fontsize=18, fontproperties=font)
        plt.draw()
        start_time = time.time()
        plt.pause(2)  # 显示2秒
        
        # 清空图像，等待用户输入
        plt.clf()
        ax = fig.add_subplot(111)
        ax.axis("off")
        fig.suptitle("请使用方向键(←↑↓→)回答或空格表示未知方向", fontsize=20, color='red', fontproperties=font)
        plt.draw()
        
        # 等待用户输入
        user_key = None
        response_time = -1
        
        def on_key(event):
            nonlocal user_key, response_time
            if event.key in ['left', 'right', 'up', 'down', ' ']:
                user_key = event.key
                response_time = time.time() - start_time
        
        # 连接键盘事件
        cid = fig.canvas.mpl_connect('key_press_event', on_key)
        
        # 等待用户输入或超时
        waiting_start = time.time()
        while user_key is None and time.time() - waiting_start < 10:  # 最多等待10秒
            plt.pause(0.1)
        
        # 清理事件连接
        fig.canvas.mpl_disconnect(cid)
        
        # 处理用户响应
        correct_direction = pair['b2a_direction'].lower()
        user_direction = key_to_direction(user_key) if user_key else 'timeout'
        is_correct = (user_direction == correct_direction)
        
        # 记录结果
        results.append({
            'camera_uuid': pair['camera_uuid'],
            'room': pair['room'],
            'frame_num_a': pair['frame_num_a'],
            'frame_num_b': pair['frame_num_b'],
            'b2a_direction': correct_direction,
            'user_answer': user_direction,
            'is_correct': is_correct,
            'response_time': response_time,
            'difficulty': pair['difficulty']
        })
        
        # 在同一窗口显示结果
        plt.clf()
        ax = fig.add_subplot(111)
        ax.axis('off')
        result_color = 'green' if is_correct else 'red'
        result_text = '正确' if is_correct else '错误'
        ax.text(0.5, 0.7, f"您的回答: {user_direction}", fontsize=16, ha='center', transform=ax.transAxes, fontproperties=font)
        ax.text(0.5, 0.5, f"结果: {result_text}", color=result_color, fontsize=20, ha='center', weight='bold', transform=ax.transAxes, fontproperties=font)
        ax.text(0.5, 0.3, f"正确答案: {correct_direction}", fontsize=16, ha='center', transform=ax.transAxes, fontproperties=font)
        plt.draw()
        plt.pause(2)
    
    # 测试完成后关闭窗口
    plt.close(fig)
    plt.ioff()  # 关闭交互模式
    
    save_results(results, output_path, participant_name)
    print(f"测试完成! 共测试 {len(results)} 对图像。")

if __name__ == "__main__":

    csv_path = "ImagePairsFromPano/5classes_dataset/test.csv"
    imgs_dir = "ImagePairsFromPano/5classes_dataset/rgb"
    num_samples = 50
    output_path = "./SpatialCognitionTest/Results"

    run_test(csv_path, imgs_dir, num_samples, output_path)
