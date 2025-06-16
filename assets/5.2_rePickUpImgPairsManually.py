'''
从合并后的差异化样本中，再筛选一批更平衡的样本，并逐一展示筛选的结果。
如对于展示的结果不满意，则可以选择剔除并补充近似类型的样本。
'''

import os
import csv
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from collections import Counter, defaultdict
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import shutil

font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14) # 中文字体

@dataclass
class SampleRequirement:
    """样本筛选要求配置"""
    total_samples: int = 105  # 总样本数
    direction_ratio: Dict[str, float] = None  # Up, Down, Left, Right 各25%
    difficulty_ratio: Dict[str, float] = None  # Easy: 20%, Normal: 40%, Hard: 40%
    
    def __post_init__(self):
        """初始化样本要求"""
        if self.direction_ratio is None:
            self.direction_ratio = {'Up': 0.25, 'Down': 0.25, 'Left': 0.25, 'Right': 0.25}
        if self.difficulty_ratio is None:
            self.difficulty_ratio = {'Easy': 0.2, 'Normal': 0.4, 'Hard': 0.4}

class BalancedSampleSelector:
    """平衡样本筛选器"""
    
    def __init__(self, csv_path: str, imgs_dir: str, output_path: str, requirements: SampleRequirement = None, organize_to_folders: bool = False):
        self.csv_path = csv_path
        self.imgs_dir = imgs_dir
        self.output_path = output_path
        self.requirements = requirements or SampleRequirement()
        self.df = pd.read_csv(csv_path)
        self.selected_samples = []
        self.rejected_samples = []
        self.organize_to_folders = organize_to_folders
        
        # 预处理数据
        self._preprocess_data()
        
    def _preprocess_data(self):
        """预处理数据，计算目标分布"""
        # 计算各类别目标数量
        self.target_counts = {}
        for direction, ratio in self.requirements.direction_ratio.items():
            self.target_counts[direction] = {}
            direction_total = int(self.requirements.total_samples * ratio)
            
            for difficulty, diff_ratio in self.requirements.difficulty_ratio.items():
                count = int(direction_total * diff_ratio)
                self.target_counts[direction][difficulty] = count
        
        # 确保总数一致
        total_allocated = sum(sum(counts.values()) for counts in self.target_counts.values())
        if total_allocated != self.requirements.total_samples:
            # 平均分配到四个方向的Hard类别
            diff = self.requirements.total_samples - total_allocated
            directions = ['Up', 'Down', 'Left', 'Right']
            for i in range(abs(diff)):
                d = directions[i % 4]
                if diff > 0:
                    self.target_counts[d]['Hard'] += 1
                elif diff < 0 and self.target_counts[d]['Hard'] > 0:
                    self.target_counts[d]['Hard'] -= 1
        
        print("目标分布:")
        for direction in ['Up', 'Down', 'Left', 'Right']:
            for difficulty in ['Easy', 'Normal', 'Hard']:
                count = self.target_counts[direction][difficulty]
                print(f"  {direction}-{difficulty}: {count}")
        print(f"总计: {sum(sum(counts.values()) for counts in self.target_counts.values())}")
        
    def _get_image_paths(self, row):
        """根据样本信息生成图片路径"""
        camera_uuid = row["camera_uuid"]
        room = row["room"]
        frame_num_a = row["frame_num_a"]
        frame_num_b = row["frame_num_b"]
        augmentation_method = row.get("augmentation_method", "none")
        
        # 根据约定格式生成图片文件名
        img_a_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_a}_{augmentation_method}.png"
        img_b_name = f"perspective_{camera_uuid}_{room[:-2]}_frame_{frame_num_b}_{augmentation_method}.png"
        
        img_a_path = os.path.join(self.imgs_dir, img_a_name)
        img_b_path = os.path.join(self.imgs_dir, img_b_name)
        
        return img_a_path, img_b_path
   
    def _check_images_exist(self, row):
        """检查图片是否存在"""
        img_a_path, img_b_path = self._get_image_paths(row)
        return os.path.exists(img_a_path) and os.path.exists(img_b_path)
    
    def auto_select_balanced_samples(self) -> List[int]:
        """自动筛选平衡样本"""
        # 按类别分组
        grouped = defaultdict(list)
        for idx, row in self.df.iterrows():
            if not self._check_images_exist(row):
                continue
            direction = row['b2a_direction']
            difficulty = row['difficulty']
            grouped[(direction, difficulty)].append(idx)
        
        selected_indices = []
        shortage_records = []  # 记录不足的类别
        
        # 按目标分布筛选
        for direction in ['Up', 'Down', 'Left', 'Right']:
            for difficulty in ['Easy', 'Normal', 'Hard']:
                target_count = self.target_counts[direction][difficulty]
                available_samples = grouped[(direction, difficulty)]
                
                print(f"{direction}-{difficulty}: 需要{target_count}个，可用{len(available_samples)}个")
                
                if len(available_samples) >= target_count:
                    # 随机选择
                    selected = random.sample(available_samples, target_count)
                    selected_indices.extend(selected)
                else:
                    # 不够的话全选，并记录不足
                    selected = available_samples
                    shortage = target_count - len(available_samples)
                    shortage_records.append((direction, difficulty, shortage))
                    selected_indices.extend(selected)
                    print(f"  警告: {direction}-{difficulty} 样本不足，仅选择{len(selected)}个，缺少{shortage}个")
        
        # 进行智能补全
        if shortage_records:
            print(f"\n开始智能补全，共需补全{sum(record[2] for record in shortage_records)}个样本")
            selected_indices = self._smart_supplement(selected_indices, grouped, shortage_records)
        
        print(f"自动筛选完成，共选择{len(selected_indices)}个样本")
        return selected_indices
    
    def _smart_supplement(self, selected_indices: List[int], grouped: Dict, shortage_records: List[Tuple]) -> List[int]:
        """智能补全样本"""
        for direction, difficulty, shortage_count in shortage_records:
            print(f"\n补全 {direction}-{difficulty}，需要{shortage_count}个样本")
            
            # 获取已选样本集合，避免重复
            selected_set = set(selected_indices)
            candidates = []
            
            # 1. 同类别不同难度的样本，Hard > Normal > Easy
            priority_difficulties = ['Hard', 'Normal', 'Easy']
            for diff in priority_difficulties:
                if diff != difficulty:
                    available = [idx for idx in grouped[(direction, diff)] if idx not in selected_set]
                    candidates.extend([(idx, 1) for idx in available])  # 优先级1
            
            # 2. 不同类别同难度的样本，Up, Down > Left, Right
            priority_directions = ['Up', 'Down', 'Left', 'Right']
            for dir_name in priority_directions:
                if dir_name != direction:
                    available = [idx for idx in grouped[(dir_name, difficulty)] if idx not in selected_set]
                    priority = 2 if dir_name in ['Up', 'Down'] else 3
                    candidates.extend([(idx, priority) for idx in available])
            
            # 3. 不同类别不同难度的样本
            for dir_name in priority_directions:
                if dir_name != direction:
                    for diff in priority_difficulties:
                        if diff != difficulty:
                            available = [idx for idx in grouped[(dir_name, diff)] if idx not in selected_set]
                            priority = 4 if dir_name in ['Up', 'Down'] else 5
                            if diff == 'Hard':
                                priority += 0
                            elif diff == 'Normal':
                                priority += 1
                            else:  # Easy
                                priority += 2
                            candidates.extend([(idx, priority) for idx in available])
            
            # 按优先级排序并选择
            candidates.sort(key=lambda x: x[1])  # 按优先级排序
            
            supplement_count = min(shortage_count, len(candidates))
            if supplement_count > 0:
                supplement_samples = [idx for idx, _ in candidates[:supplement_count]]
                selected_indices.extend(supplement_samples)
                
                # 打印补全信息
                for idx in supplement_samples:
                    row = self.df.iloc[idx]
                    sup_dir = row['b2a_direction']
                    sup_diff = row['difficulty']
                    print(f"  补全样本 #{idx}: {sup_dir}-{sup_diff} -> {direction}-{difficulty}")
                
                print(f"  成功补全{supplement_count}个样本")
            else:
                print(f"  无法找到更多样本进行补全")
        
        return selected_indices
    
    def display_sample_interactive(self, idx: int, current_idx: int, total_samples: int):
        """交互式显示单个样本"""
        row = self.df.iloc[idx]
        img_a_path, img_b_path = self._get_image_paths(row)
        
        if not (os.path.exists(img_a_path) and os.path.exists(img_b_path)):
            print(f"图片不存在: {img_a_path} 或 {img_b_path}")
            return None
        
        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")
        
        # 创建显示窗口
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        axes[0].imshow(img_a)
        axes[0].set_title("Image A", fontsize=14, fontproperties=font)
        axes[0].axis("off")
        
        axes[1].imshow(img_b)
        axes[1].set_title("Image B", fontsize=14, fontproperties=font)
        axes[1].axis("off")
        
        # 显示样本信息和操作提示
        direction = row['b2a_direction']
        difficulty = row['difficulty']
        aug_method = row['augmentation_method']
        room = row['room']
        
        title = f"样本 #{idx}: {direction} - {difficulty} | 进度: {current_idx}/{total_samples}\n"
        title += f"房间: {room}, 增强: {aug_method}\n"
        title += f"帧: {row['frame_num_a']} → {row['frame_num_b']}\n\n"
        title += "按键操作: [K]保留 [R]替换 [S]跳过 [Q]完成筛选 [ESC]退出"
        
        plt.suptitle(title, fontsize=14, fontproperties=font)
        plt.tight_layout()
        
        # 设置按键响应
        user_choice = {'action': None}
        
        def on_key_press(event):
            if event.key in ['k', 'K']:
                user_choice['action'] = 'keep'
                plt.close(fig)
            elif event.key in ['r', 'R']:
                user_choice['action'] = 'replace'
                plt.close(fig)
            elif event.key in ['s', 'S']:
                user_choice['action'] = 'skip'
                plt.close(fig)
            elif event.key in ['q', 'Q']:
                user_choice['action'] = 'quit'
                plt.close(fig)
            elif event.key == 'escape':
                user_choice['action'] = 'exit'
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # 设置窗口为可聚焦
        fig.canvas.setWindowTitle(f"样本筛选 - 样本 #{idx}")
        
        print(f"\n进度: {current_idx}/{total_samples}")
        print(f"当前样本: #{idx} | {direction}-{difficulty} | {room} | {aug_method}")
        print("请在图片窗口中按键选择: [K]保留 [R]替换 [S]跳过 [Q]完成筛选 [ESC]退出")
        
        plt.show()
        
        return user_choice['action']
    
    def display_replacement_sample(self, idx: int, option_num: int, total_options: int):
        """显示替换选项样本"""
        row = self.df.iloc[idx]
        img_a_path, img_b_path = self._get_image_paths(row)
        
        if not (os.path.exists(img_a_path) and os.path.exists(img_b_path)):
            print(f"图片不存在: {img_a_path} 或 {img_b_path}")
            return None
        
        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")
        
        # 创建显示窗口
        fig, axes = plt.subplots(1, 2, figsize=(14, 8))
        
        axes[0].imshow(img_a)
        axes[0].set_title("Image A", fontsize=14, fontproperties=font)
        axes[0].axis("off")
        
        axes[1].imshow(img_b)
        axes[1].set_title("Image B", fontsize=14, fontproperties=font)
        axes[1].axis("off")
        
        # 显示样本信息和操作提示
        direction = row['b2a_direction']
        difficulty = row['difficulty']
        aug_method = row['augmentation_method']
        room = row['room']
        
        title = f"替换选项 {option_num}/{total_options} - 样本 #{idx}: {direction} - {difficulty}\n"
        title += f"房间: {room}, 增强: {aug_method}\n"
        title += f"帧: {row['frame_num_a']} → {row['frame_num_b']}\n\n"
        title += "按键操作: [Y]确认替换 [N]查看下一个 [C]取消替换"
        
        plt.suptitle(title, fontsize=14, fontproperties=font)
        plt.tight_layout()
        
        # 设置按键响应
        user_choice = {'action': None}
        
        def on_key_press(event):
            if event.key in ['y', 'Y']:
                user_choice['action'] = 'confirm'
                plt.close(fig)
            elif event.key in ['n', 'N']:
                user_choice['action'] = 'next'
                plt.close(fig)
            elif event.key in ['c', 'C']:
                user_choice['action'] = 'cancel'
                plt.close(fig)
            elif event.key == 'escape':
                user_choice['action'] = 'cancel'
                plt.close(fig)
        
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        fig.canvas.setWindowTitle(f"替换选项 - 样本 #{idx}")
        
        print(f"\n替换选项 {option_num}/{total_options} - 样本 #{idx}")
        print(f"  {direction}-{difficulty} | {room} | {aug_method}")
        print("请在图片窗口中按键选择: [Y]确认替换 [N]查看下一个 [C]取消替换")
        
        plt.show()
        
        return user_choice['action']
    
    def display_samples_grid(self, indices: List[int], cols: int = 5, save_path: Optional[str] = None):
        """网格显示多个样本"""
        if not indices:
            print("没有样本需要显示")
            return
        
        rows = (len(indices) + cols - 1) // cols
        fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 3, rows * 6))
        
        if rows == 1:
            axes = axes.reshape(2, -1)
        
        for i, idx in enumerate(indices):
            if i >= rows * cols:
                break
                
            row_idx = i // cols
            col_idx = i % cols
            
            row = self.df.iloc[idx]
            img_a_path, img_b_path = self._get_image_paths(row)
            
            if not (os.path.exists(img_a_path) and os.path.exists(img_b_path)):
                continue
            
            img_a = Image.open(img_a_path).convert("RGB")
            img_b = Image.open(img_b_path).convert("RGB")
            
            # 统一尺寸
            target_size = (256, 256)
            img_a = img_a.resize(target_size)
            img_b = img_b.resize(target_size)
            
            # 显示图片A
            axes[row_idx * 2, col_idx].imshow(img_a)
            axes[row_idx * 2, col_idx].set_title(f"#{idx}-A\n{row['b2a_direction']}-{row['difficulty']}", fontsize=10, fontproperties=font)
            axes[row_idx * 2, col_idx].axis("off")
            
            # 显示图片B
            axes[row_idx * 2 + 1, col_idx].imshow(img_b)
            axes[row_idx * 2 + 1, col_idx].set_title(f"#{idx}-B", fontsize=10, fontproperties=font)
            axes[row_idx * 2 + 1, col_idx].axis("off")
        
        # 隐藏多余的子图
        for i in range(len(indices), rows * cols):
            row_idx = i // cols
            col_idx = i % cols
            axes[row_idx * 2, col_idx].axis("off")
            axes[row_idx * 2 + 1, col_idx].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            print(f"网格图已保存到: {save_path}")
        
        # plt.show()
    
    def get_current_distribution(self, indices: List[int]) -> Dict:
        """获取当前选择样本的分布"""
        dist = defaultdict(lambda: defaultdict(int))
        
        for idx in indices:
            row = self.df.iloc[idx]
            direction = row['b2a_direction']
            difficulty = row['difficulty']
            dist[direction][difficulty] += 1
        
        return dist
    
    def print_distribution_comparison(self, indices: List[int]):
        """打印分布对比"""
        current_dist = self.get_current_distribution(indices)
        
        print("\n=== 分布对比 ===")
        print(f"{'类别':<15} {'目标':<8} {'当前':<8} {'差异':<8}")
        print("-" * 45)
        
        total_target = 0
        total_current = 0
        
        for direction in ['Up', 'Down', 'Left', 'Right']:
            for difficulty in ['Easy', 'Normal', 'Hard']:
                target = self.target_counts[direction][difficulty]
                current = current_dist[direction][difficulty]
                diff = current - target
                
                total_target += target
                total_current += current
                
                status = "✓" if diff == 0 else ("+" if diff > 0 else "-")
                print(f"{direction}-{difficulty:<8} {target:<8} {current:<8} {diff:+3d} {status}")
        
        print("-" * 45)
        print(f"{'总计':<15} {total_target:<8} {total_current:<8} {total_current - total_target:+3d}")
    
    def find_similar_samples(self, reference_idx: int, direction: str, difficulty: str, limit: int = 5) -> List[int]:
        """查找相似样本用于替换"""
        reference_row = self.df.iloc[reference_idx]
        reference_room = reference_row['room']
        reference_aug = reference_row['augmentation_method']
        
        candidates = []
        
        # 1. 首先查找完全匹配的样本（同类别同难度）
        for idx, row in self.df.iterrows():
            if idx == reference_idx or idx in self.selected_samples:
                continue
            if row['b2a_direction'] == direction and row['difficulty'] == difficulty:
                if not self._check_images_exist(row):
                    continue
                
                # 计算相似度分数
                score = 0
                if row['room'] == reference_room:
                    score += 2
                if row['augmentation_method'] == reference_aug:
                    score += 1
                
                candidates.append((idx, score, 0))  # 优先级0（最高）
        
        # 如果完全匹配的样本不够，按照优先级补全
        if len(candidates) < limit:
            # 获取当前已有的样本索引
            existing_indices = {idx for idx, _, _ in candidates}
            existing_indices.add(reference_idx)
            existing_indices.update(self.selected_samples)
            
            # 2. 同类别不同难度的样本，Hard > Normal > Easy
            priority_difficulties = ['Hard', 'Normal', 'Easy']
            for i, diff in enumerate(priority_difficulties):
                if diff != difficulty:
                    for idx, row in self.df.iterrows():
                        if idx in existing_indices:
                            continue
                        if row['b2a_direction'] == direction and row['difficulty'] == diff:
                            if not self._check_images_exist(row):
                                continue
                            
                            score = 0
                            if row['room'] == reference_room:
                                score += 2
                            if row['augmentation_method'] == reference_aug:
                                score += 1
                            
                            priority = 1 + i  # Hard=1, Normal=2, Easy=3
                            candidates.append((idx, score, priority))
                            existing_indices.add(idx)
            
            # 3. 不同类别同难度的样本，Up, Down > Left, Right
            priority_directions = ['Up', 'Down', 'Left', 'Right']
            for dir_name in priority_directions:
                if dir_name != direction:
                    for idx, row in self.df.iterrows():
                        if idx in existing_indices:
                            continue
                        if row['b2a_direction'] == dir_name and row['difficulty'] == difficulty:
                            if not self._check_images_exist(row):
                                continue
                            
                            score = 0
                            if row['room'] == reference_room:
                                score += 2
                            if row['augmentation_method'] == reference_aug:
                                score += 1
                            
                            priority = 4 if dir_name in ['Up', 'Down'] else 5
                            candidates.append((idx, score, priority))
                            existing_indices.add(idx)
            
            # 4. 不同类别不同难度的样本
            for dir_name in priority_directions:
                if dir_name != direction:
                    for i, diff in enumerate(priority_difficulties):
                        if diff != difficulty:
                            for idx, row in self.df.iterrows():
                                if idx in existing_indices:
                                    continue
                                if row['b2a_direction'] == dir_name and row['difficulty'] == diff:
                                    if not self._check_images_exist(row):
                                        continue
                                    
                                    score = 0
                                    if row['room'] == reference_room:
                                        score += 2
                                    if row['augmentation_method'] == reference_aug:
                                        score += 1
                                    
                                    base_priority = 6 if dir_name in ['Up', 'Down'] else 7
                                    priority = base_priority + i  # Hard=0, Normal=1, Easy=2
                                    candidates.append((idx, score, priority))
                                    existing_indices.add(idx)
        
        # 按优先级和相似度排序
        candidates.sort(key=lambda x: (x[2], -x[1]))  # 先按优先级升序，再按相似度降序
        
        return [idx for idx, _, _ in candidates[:limit]]
    
    def interactive_selection(self):
        """交互式样本筛选 - 逐个展示样本"""
        print("开始交互式样本筛选...")
        
        if self.selected_samples is None: # 如果没有预选样本，则自动选择平衡样本
            selected_indices = self.auto_select_balanced_samples()
            self.selected_samples = selected_indices.copy()
        
        print(f"\n开始逐个展示{len(self.selected_samples)}个样本，请在图片窗口中做出选择...")
        print("=" * 60)
        
        # 逐个展示样本
        final_samples = []
        for i, idx in enumerate(self.selected_samples):
            print(f"\n进度: {i+1}/{len(self.selected_samples)}")
            
            # 显示当前样本
            row = self.df.iloc[idx]
            direction = row['b2a_direction']
            difficulty = row['difficulty']
            
            print(f"当前样本: #{idx} | {direction}-{difficulty} | {row['room']} | {row['augmentation_method']}")
            
            # 使用交互式显示
            action = self.display_sample_interactive(idx, i+1, len(self.selected_samples))
            
            if action == 'keep':
                # 保留当前样本
                final_samples.append(idx)
                print(f"✓ 已保留样本 #{idx}")
            elif action == 'replace':
                # 替换样本
                replacement = self._find_and_select_replacement(idx, direction, difficulty)
                if replacement is not None:
                    final_samples.append(replacement)
                    print(f"✓ 已用样本 #{replacement} 替换样本 #{idx}")
                else:
                    print("未找到合适替换，保留原样本")
                    final_samples.append(idx)
            elif action == 'skip':
                # 跳过当前样本
                print(f"× 已跳过样本 #{idx}")
            elif action == 'quit':
                # 完成筛选
                final_samples.append(idx)
                print(f"✓ 已保留样本 #{idx}")
                print("提前完成筛选")
                self.selected_samples = final_samples
                self._finalize_selection()
                return
            elif action == 'exit':
                # 退出程序
                print("用户退出程序")
                return
        
        # 更新最终选择
        self.selected_samples = final_samples
        self._finalize_selection()
    
    def _find_and_select_replacement(self, old_idx: int, direction: str, difficulty: str) -> Optional[int]:
        """查找并选择替换样本"""
        # 查找相似样本
        similar_samples = self.find_similar_samples(old_idx, direction, difficulty, limit=10)
        
        if not similar_samples:
            print(f"没有找到 {direction}-{difficulty} 类型的相似样本")
            return None
        
        print(f"\n找到 {len(similar_samples)} 个 {direction}-{difficulty} 类型的相似样本")
        print("将逐个展示替换选项，请在图片窗口中选择...")
        
        # 逐个展示替换选项
        for i, idx in enumerate(similar_samples):
            row = self.df.iloc[idx]
            print(f"\n替换选项 {i+1}/{len(similar_samples)}: 样本 #{idx} - {row['room']}, {row['augmentation_method']}")
            
            action = self.display_replacement_sample(idx, i+1, len(similar_samples))
            
            if action == 'confirm':
                print(f"✓ 选择替换样本 #{idx}")
                return idx
            elif action == 'next':
                print("查看下一个替换选项...")
                continue
            elif action == 'cancel':
                print("取消替换")
                return None
        
        print("已查看所有替换选项，未选择替换")
        return None
    
    def _finalize_selection(self):
        """完成筛选，显示最终结果并保存"""
        print(f"\n筛选完成！最终选择了 {len(self.selected_samples)} 个样本")
        
        # 显示分布统计
        self.print_distribution_comparison(self.selected_samples)
        
        # 直接保存结果
        print("\n自动保存筛选结果...")
        self._save_selection()
    
    def _save_selection(self):
        """保存当前选择"""
        if not self.selected_samples:
            print("没有选择任何样本")
            return
        
        # 创建输出DataFrame
        selected_df = self.df.iloc[self.selected_samples].copy()
        selected_df['selection_index'] = range(1, len(selected_df) + 1)
        
        # 保存到CSV
        output_path = os.path.join(self.output_path, f"balanced_samples_{self.requirements.total_samples}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        selected_df.to_csv(output_path, index=False)
        
        # 保存网格图
        grid_path = os.path.join(self.output_path, f"balanced_samples_{self.requirements.total_samples}_grid.png")
        self.display_samples_grid(self.selected_samples, save_path=grid_path)
        
        print(f"选择的样本已保存到: {output_path}")
        print(f"网格图已保存到: {grid_path}")
        
        # 保存统计报告
        self._save_statistics_report(selected_df, output_path.replace('.csv', '_statistics.txt'))
        
        # 可选：将图片复制到分层文件夹
        if self.organize_to_folders:
            print("正在将样本图片复制到分层文件夹...")
            for idx in self.selected_samples:
                row = self.df.iloc[idx]
                direction = row['b2a_direction'].lower()
                room = row['room']
                frame_a = row['frame_num_a']
                frame_b = row['frame_num_b']
                aug = row.get('augmentation_method', 'none')
                # 三级文件夹名
                subfolder = f"{room}_{frame_a}_{frame_b}"
                target_dir = os.path.join(self.output_path, "organized_samples", direction, subfolder)
                os.makedirs(target_dir, exist_ok=True)
                # 源图片路径
                img_a_path, img_b_path = self._get_image_paths(row)
                # 目标图片名
                img_a_name = os.path.basename(img_a_path)
                img_b_name = os.path.basename(img_b_path)
                # 复制图片
                if os.path.exists(img_a_path):
                    shutil.copy(img_a_path, os.path.join(target_dir, img_a_name))
                if os.path.exists(img_b_path):
                    shutil.copy(img_b_path, os.path.join(target_dir, img_b_name))
            print("图片已按层次结构复制完成。")
    
    def _save_statistics_report(self, df: pd.DataFrame, report_path: str):
        """保存统计报告"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("平衡样本筛选统计报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本信息
            f.write(f"总样本数: {len(df)}\n")
            f.write(f"目标样本数: {self.requirements.total_samples}\n\n")
            
            # 分布统计
            f.write("类别分布:\n")
            dist = self.get_current_distribution(df.index.tolist())
            for direction in ['Up', 'Down', 'Left', 'Right']:
                f.write(f"  {direction}:\n")
                for difficulty in ['Easy', 'Normal', 'Hard']:
                    target = self.target_counts[direction][difficulty]
                    current = dist[direction][difficulty]
                    f.write(f"    {difficulty}: {current}/{target}\n")
                f.write("\n")
            
            # 增强方法分布
            f.write("增强方法分布:\n")
            aug_dist = df['augmentation_method'].value_counts()
            for aug, count in aug_dist.items():
                f.write(f"  {aug}: {count}\n")
            f.write("\n")
            
            # 房间分布
            f.write("房间分布:\n")
            room_dist = df['room'].value_counts()
            for room, count in room_dist.items():
                f.write(f"  {room}: {count}\n")
        
        print(f"统计报告已保存到: {report_path}")

def main():
    """主函数"""
    csv_path = "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance_>3/combined_sample_results.csv"
    imgs_dir = "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance_>3/copied_imgs_withLabel_all"
    output_path = "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance_>3"
    organized_samples = True  # 是否将样本图片按类别和房间组织到分层文件夹
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: CSV文件不存在: {csv_path}")
        return
    
    if not os.path.exists(imgs_dir):
        print(f"错误: 图片目录不存在: {imgs_dir}")
        return
    
    os.makedirs(imgs_dir, exist_ok=True)
    
    # 设置随机种子以便重现结果
    # random.seed(42)
    # np.random.seed(42)
    
    # 创建筛选器
    requirements = SampleRequirement()
    # requirements.total_samples = total_samples
    selector = BalancedSampleSelector(csv_path, imgs_dir, output_path, requirements, organized_samples)
    
    print("平衡样本筛选工具")
    print("=" * 30)
    print(f"数据来源: {csv_path}")
    print(f"图片目录: {imgs_dir}")
    print(f"输出目录: {output_path}")
    print(f"总样本数: {len(selector.df)}")

    
    # 自动筛选并询问用户
    auto_indices = selector.auto_select_balanced_samples()
    selector.selected_samples = auto_indices.copy()
    print(f"\n自动筛选共选择 {len(auto_indices)} 个样本。\n")
    use_auto = input("是否直接使用自动筛选的样本并保存结果？(y/n): ").strip().lower()
    if use_auto == 'y':
        selector._finalize_selection()
    else:
        print("\n操作说明:")
        print("- 将逐个展示自动筛选的样本")
        print("- 对每个样本在图片窗口中按键选择: [K]保留 [R]替换 [S]跳过 [Q]完成筛选")
        print("- 替换时按键选择: [Y]确认替换 [N]查看下一个 [C]取消替换")
        print("- 按键时请确保图片窗口处于激活状态")
        selector.interactive_selection()

if __name__ == "__main__":
    main()
