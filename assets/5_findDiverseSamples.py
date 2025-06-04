'''
寻找使模型间差异化最大的样本
'''
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter, defaultdict
import numba
from tqdm import tqdm
import os

# 将Numba函数定义在类外部
@numba.jit(nopython=True)
def build_matrix_numba(all_softmax_data: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    """构建模型预测矩阵 (Numba加速)"""
    M, _, _ = all_softmax_data.shape
    N = len(sample_indices)
    
    matrix = np.zeros((M, 4*N), dtype=np.float64)
    
    for i in range(M):
        for j in range(N):
            idx = sample_indices[j]
            softmax_values = all_softmax_data[i, idx, :4]  # 只取前4个类别
            matrix[i, j*4:(j+1)*4] = softmax_values
            
    return matrix

class SampleSelector:
    """重构后的样本选择器"""
    
    def __init__(self, lambda_param: float = 0.1, strict_accuracy: bool = True, 
                 accuracy_tolerance: float = 0.15):
        self.lambda_param = lambda_param
        self.strict_accuracy = strict_accuracy
        self.accuracy_tolerance = accuracy_tolerance  # 可调节的容忍度
        self.difficulty_map_encode = {'Easy': 0, 'Normal': 1, 'Hard': 2, 'Unknown': -1}
        self.selected_samples = set()  # 用于唯一性约束
        
    def load_and_merge_model_data(self, model_paths: List[List[str]]) -> List[pd.DataFrame]:
        """
        加载并合并每个模型下不同数据增强方法的CSV
        
        Args:
            model_paths: 二维列表，第一维是模型，第二维是该模型下不同增强方法的CSV路径
            
        Returns:
            每个模型合并后的DataFrame列表
        """
        merged_models = []
        
        for model_idx, augmentation_paths in enumerate(model_paths):
            model_dfs = []
            
            for aug_path in augmentation_paths:
                try:
                    df = pd.read_csv(aug_path)
                    # 从路径中提取增强方法名
                    aug_method = self._extract_augmentation_method(aug_path)
                    df['augmentation_method'] = aug_method
                    df['model_id'] = model_idx
                    model_dfs.append(df)
                except Exception as e:
                    print(f"警告: 加载文件 {aug_path} 失败: {e}")
                    continue
            
            if model_dfs:
                # 合并同一模型下的所有增强方法数据
                merged_model_df = pd.concat(model_dfs, ignore_index=True)
                merged_models.append(merged_model_df)
            else:
                print(f"警告: 模型 {model_idx} 没有成功加载任何数据")
        
        return merged_models
    
    def _extract_augmentation_method(self, file_path: str) -> str:
        """从文件路径中提取增强方法名"""
        path_parts = file_path.split('/')
        
        # 查找包含增强方法的路径部分
        for part in reversed(path_parts):
            if part in ['none', 'edge_reduction', 'texture_reduction', 'gray']:
                return part
    
        # 如果没找到，从文件名中提取
        filename = os.path.basename(file_path)
        if 'none' in filename:
            return 'none'
        elif 'gray' in filename:
            return 'gray'
        elif 'edge_reduction' in filename:
            return 'edge_reduction'
        elif 'texture_reduction' in filename:
            return 'texture_reduction'
        else:
            return 'unknown'
    
    def _build_matrix_numba(self, all_softmax_data: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
        """构建模型预测矩阵 (调用Numba加速函数)"""
        return build_matrix_numba(all_softmax_data, sample_indices)
    
    def _prepare_data(self, model_dfs: List[pd.DataFrame], label_filter: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[int], Dict]:
        """预处理数据"""
        # 使用第一个模型的数据作为基准来获取样本索引
        base_df = model_dfs[0]
        
        # 按标签过滤
        if label_filter is not None:
            initial_indices = base_df[base_df['label'] == label_filter].index.tolist()
        else:
            initial_indices = list(range(len(base_df)))
        
        print(f"标签过滤后的初始样本数: {len(initial_indices)}")
        
        # 构建所有模型的softmax数据
        M = len(model_dfs)
        total_samples = len(base_df)
        all_softmax_data = np.zeros((M, total_samples, 4), dtype=np.float64)
        
        for i, df in enumerate(model_dfs):
            softmax_cols = [f'softmax_{k}' for k in range(4)]
            all_softmax_data[i, :, :] = df[softmax_cols].values
        
        # 提取难度信息
        all_difficulties = np.array([
            self.difficulty_map_encode.get(d, -1) 
            for d in base_df['difficulty']
        ], dtype=np.int64)
        
        # 构建样本元信息字典
        sample_info = {}
        id_cols = ['camera_uuid', 'room', 'frame_num_a', 'frame_num_b']
        for idx in initial_indices:
            key = tuple(base_df.iloc[idx][id_cols].values)
            sample_info[idx] = {
                'key': key,
                'label': base_df.iloc[idx]['label'],
                'difficulty': base_df.iloc[idx]['difficulty'],
                'augmentation_method': base_df.iloc[idx]['augmentation_method']
            }
        
        return all_softmax_data, all_difficulties, initial_indices, sample_info
    
    def _check_all_constraints(self, 
                              candidate_idx: int,
                              current_selection: List[int],
                              all_softmax_data: np.ndarray,
                              all_difficulties: np.ndarray,
                              sample_info: Dict,
                              label_filter: Optional[int],
                              target_difficulty_counts: Dict[int, int],
                              current_difficulty_counts: Dict[int, int],
                              target_augmentation_counts: Dict[str, int],
                              current_augmentation_counts: Dict[str, int]) -> Tuple[bool, float]:
        """
        检查所有约束条件并返回是否满足以及目标函数值
        
        Returns:
            (is_valid, objective_score)
        """
        # 1. 唯一性约束：检查是否已有相同的图像对
        candidate_key = sample_info[candidate_idx]['key']
        if candidate_key in self.selected_samples:
            return False, -float('inf')
        
        # 2. Unknown结果约束：排除Unknown难度
        if all_difficulties[candidate_idx] == -1:
            return False, -float('inf')
        
        # 构建试验性选择
        trial_selection = current_selection + [candidate_idx]
        trial_selection_np = np.array(trial_selection, dtype=np.int64)
        
        # 3. 矩阵奇异值约束（结果差异约束）
        try:
            A_trial = self._build_matrix_numba(all_softmax_data, trial_selection_np)
            singular_values = np.linalg.svd(A_trial, compute_uv=False)
            
            if singular_values[-1] < 1e-10:
                return False, -float('inf')
            
            sigma_min = singular_values[-1]
            sigma_max = singular_values[0]
            diversity_score = sigma_min - self.lambda_param * (sigma_max / sigma_min - 1)
            
        except np.linalg.LinAlgError:
            return False, -float('inf')
        
        # 4. 动态调整的正确率约束
        if label_filter is not None:
            M = all_softmax_data.shape[0]
            pred_labels = np.argmax(all_softmax_data[:, trial_selection, :4], axis=2)
            is_correct = (pred_labels == label_filter)
            
            # 检查是否有全对或全错的样本
            all_correct_samples = np.all(is_correct, axis=0)
            all_incorrect_samples = np.all(~is_correct, axis=0)
            
            if np.any(all_correct_samples) or np.any(all_incorrect_samples):
                return False, -float('inf')
            
            # 计算各模型的正确率
            model_accuracies = np.mean(is_correct, axis=1)
            
            # 根据当前样本数量动态调整约束严格程度
            num_samples = len(trial_selection)
            
            if self.strict_accuracy:
                # 动态调整容忍度：样本数量越少，容忍度越大
                if num_samples <= 3:
                    # 初期阶段：非常宽松
                    dynamic_tolerance = 0.6  # 80%的容忍度
                elif num_samples <= 6:
                    # 早期阶段：较宽松
                    dynamic_tolerance = 0.3  # 50%的容忍度
                elif num_samples <= 10:
                    # 中期阶段：逐渐严格
                    dynamic_tolerance = 0.2  # 30%的容忍度
                else:
                    # 后期阶段：严格约束
                    dynamic_tolerance = self.accuracy_tolerance  # 使用设定值
                
                # 计算约束指标
                accuracy_std = np.std(model_accuracies)
                accuracy_range = np.max(model_accuracies) - np.min(model_accuracies)
                mean_accuracy = np.mean(model_accuracies)
                max_deviation = np.max(np.abs(model_accuracies - mean_accuracy))
                
                # 应用动态约束
                if (accuracy_std > dynamic_tolerance or 
                    accuracy_range > dynamic_tolerance * 2 or 
                    max_deviation > dynamic_tolerance * 1.5):
                    
                    # 如果是初期阶段，给予惩罚而不是直接拒绝
                    if num_samples <= 5:
                        diversity_score -= accuracy_std * 2  # 惩罚但不拒绝
                    else:
                        return False, -float('inf')  # 后期直接拒绝
                
                # 奖励好的平衡（使用动态容忍度）
                if dynamic_tolerance > 0:
                    balance_score = 1.0 - accuracy_std / dynamic_tolerance
                    diversity_score += balance_score * 0.5  # 降低奖励权重
            else:
                # 宽松模式：仅惩罚
                accuracy_std = np.std(model_accuracies)
                if accuracy_std > self.accuracy_tolerance:
                    diversity_score -= accuracy_std * 4
        
        # 5. 增强方法平衡约束（提高优先级，更严格）
        candidate_augmentation = sample_info[candidate_idx]['augmentation_method']
        if candidate_augmentation in current_augmentation_counts:
            aug_count = current_augmentation_counts[candidate_augmentation] + 1
            target_aug_count = target_augmentation_counts.get(candidate_augmentation, 0)
            
            # 更严格的平衡约束
            if aug_count > target_aug_count + 1:  # 只允许+1的偏差
                diversity_score -= 2.0  # 增加惩罚力度
            elif aug_count > target_aug_count:
                diversity_score -= 0.5  # 轻微惩罚
            else:
                # 奖励不足的增强方法
                diversity_score += 0.2
        
        # 6. 难度平衡约束（降低优先级）
        candidate_difficulty = all_difficulties[candidate_idx]
        if candidate_difficulty in current_difficulty_counts:
            new_count = current_difficulty_counts[candidate_difficulty] + 1
            target_count = target_difficulty_counts.get(candidate_difficulty, 0)
            
            # 如果超出目标太多，给予轻微惩罚
            if new_count > target_count + 2:
                diversity_score -= 1.0  # 降低惩罚力度
        
        return True, diversity_score
    
    def select_diverse_samples(self,
                              model_paths: List[List[str]],
                              N: int,
                              label_filter: Optional[int] = None,
                              balance_difficulty: bool = True,
                              balance_augmentation: bool = True) -> Tuple[List[int], np.ndarray, pd.DataFrame]:
        """
        主要的样本选择函数，一次性实现所有约束
        
        Args:
            model_paths: 二维列表，第一维是模型，第二维是该模型下不同增强方法的CSV路径
            N: 要选择的样本数量
            label_filter: 标签过滤器
            balance_difficulty: 是否平衡难度
            balance_augmentation: 是否平衡增强方法
            
        Returns:
            (selected_indices, final_matrix, selected_samples_df)
        """
        print("加载并合并模型数据...")
        model_dfs = self.load_and_merge_model_data(model_paths)
        
        if not model_dfs:
            raise ValueError("没有成功加载任何模型数据")
        
        print("预处理数据...")
        all_softmax_data, all_difficulties, initial_indices, sample_info = self._prepare_data(
            model_dfs, label_filter
        )
        
        # 排除任一模型预测为类别4的样本
        pred_labels_all = np.argmax(all_softmax_data, axis=2)
        valid_mask = np.all(pred_labels_all[:, initial_indices] != 4, axis=0)
        initial_indices = [initial_indices[i] for i, v in enumerate(valid_mask) if v]
        print(f"排除预测为类别4的样本后，剩余: {len(initial_indices)}")
        
        if len(initial_indices) < N:
            raise ValueError(f"可用样本数量({len(initial_indices)})小于请求数量({N})")
        
        # 设置难度平衡目标
        if balance_difficulty:
            valid_difficulties = [0, 1, 2]  # Easy, Normal, Hard
            target_difficulty_counts = {d: N // 3 for d in valid_difficulties}
            # 处理除不尽的情况
            for i in range(N % 3):
                target_difficulty_counts[valid_difficulties[i]] += 1
        else:
            target_difficulty_counts = {}
        
        # 设置增强方法平衡目标
        if balance_augmentation:
            # 统计所有可用的增强方法
            available_augmentations = set()
            for idx in initial_indices:
                aug_method = sample_info[idx]['augmentation_method']
                available_augmentations.add(aug_method)
            
            available_augmentations = sorted(list(available_augmentations))
            num_augmentations = len(available_augmentations)
            
            target_augmentation_counts = {aug: N // num_augmentations for aug in available_augmentations}
            # 处理除不尽的情况
            for i in range(N % num_augmentations):
                target_augmentation_counts[available_augmentations[i]] += 1
                
            print(f"增强方法平衡目标: {target_augmentation_counts}")
        else:
            target_augmentation_counts = {}
            available_augmentations = []
        
        current_difficulty_counts = {0: 0, 1: 0, 2: 0}
        current_augmentation_counts = {aug: 0 for aug in available_augmentations}
        selected_indices = []
        remaining_indices = set(initial_indices)
        self.selected_samples = set()  # 重置已选样本集合
        
        print(f"开始选择 {N} 个多样化样本...")
        pbar = tqdm(range(N), desc="选择样本")
        
        for iteration in pbar:
            best_score = -float('inf')
            best_idx = None
            
            # 构建候选池：优先选择不足的类别
            candidate_pool = list(remaining_indices)
            
            # 更积极的平衡策略
            if balance_augmentation and iteration < N - 2:  # 几乎到最后才放松约束
                # 找出最需要的增强方法（严格按照不足数量排序）
                augmentation_needs = []
                for aug in available_augmentations:
                    current_count = current_augmentation_counts[aug]
                    target_count = target_augmentation_counts[aug]
                    need_level = target_count - current_count
                    if need_level > 0:
                        augmentation_needs.append((aug, need_level))
                
                # 按需求程度排序
                augmentation_needs.sort(key=lambda x: x[1], reverse=True)
                
                if augmentation_needs:
                    # 优先选择最需要的增强方法
                    most_needed_augs = [aug for aug, _ in augmentation_needs[:2]]  # 选择最需要的2种
                    
                    preferred_candidates = []
                    for idx in remaining_indices:
                        idx_augmentation = sample_info[idx]['augmentation_method']
                        if idx_augmentation in most_needed_augs:
                            preferred_candidates.append(idx)
                    
                    if preferred_candidates:
                        candidate_pool = preferred_candidates
                        print(f"\n第{iteration+1}轮: 优先选择增强方法 {most_needed_augs}")
            
            # 在候选池中寻找最佳样本
            for candidate_idx in candidate_pool:
                is_valid, score = self._check_all_constraints(
                    candidate_idx, selected_indices, all_softmax_data,
                    all_difficulties, sample_info, label_filter,
                    target_difficulty_counts, current_difficulty_counts,
                    target_augmentation_counts, current_augmentation_counts
                )
                
                if is_valid and score > best_score:
                    best_score = score
                    best_idx = candidate_idx
            
            if best_idx is not None:
                # 添加最佳样本
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                self.selected_samples.add(sample_info[best_idx]['key'])
                
                # 更新难度计数
                difficulty = all_difficulties[best_idx]
                if difficulty in current_difficulty_counts:
                    current_difficulty_counts[difficulty] += 1
                
                # 更新增强方法计数
                augmentation = sample_info[best_idx]['augmentation_method']
                if augmentation in current_augmentation_counts:
                    current_augmentation_counts[augmentation] += 1
                
                # 更新进度条信息
                pbar.set_postfix(
                    score=f"{best_score:.4f}",
                    difficulty=current_difficulty_counts,
                    augmentation=dict(current_augmentation_counts)
                )
            else:
                print(f"\n警告: 第 {iteration+1} 轮无法找到满足约束的样本，提前停止")
                break
        
        pbar.close()
        
        # 验证样本一致性
        print("验证样本一致性...")
        if not self._verify_sample_consistency(model_dfs, selected_indices):
            print("警告: 样本一致性验证失败")
        
        # 构建最终矩阵
        print("构建最终矩阵...")
        final_matrix = self._build_matrix_numba(
            all_softmax_data, np.array(selected_indices, dtype=np.int64)
        )
        
        # 收集选中样本信息
        selected_samples_df = self._collect_sample_info(model_dfs[0], selected_indices)
        
        # 输出统计信息
        self._print_statistics(selected_indices, all_difficulties, final_matrix, sample_info)
        
        return selected_indices, final_matrix, selected_samples_df
    
    def _verify_sample_consistency(self, model_dfs: List[pd.DataFrame], selected_indices: List[int]) -> bool:
        """验证选定样本在所有模型中的一致性"""
        if not selected_indices:
            return True
        
        id_cols = ['camera_uuid', 'room', 'frame_num_a', 'frame_num_b']
        
        for idx in selected_indices:
            reference_values = model_dfs[0].iloc[idx][id_cols].values
            for df in model_dfs[1:]:
                current_values = df.iloc[idx][id_cols].values
                if not all(ref == cur for ref, cur in zip(reference_values, current_values)):
                    print(f"不一致的样本索引: {idx}")
                    return False
        
        return True
    
    def _collect_sample_info(self, base_df: pd.DataFrame, selected_indices: List[int]) -> pd.DataFrame:
        """收集选中样本的详细信息"""
        if not selected_indices:
            return pd.DataFrame()
        
        info_cols = ['camera_uuid', 'room', 'frame_num_a', 'frame_num_b', 'label', 'difficulty', 'augmentation_method']
        selected_samples_info = []
        
        for i, idx in enumerate(selected_indices):
            sample_data = base_df.iloc[idx][info_cols].to_dict()
            sample_data['original_index'] = idx
            sample_data['selection_order'] = i + 1
            selected_samples_info.append(sample_data)
        
        df = pd.DataFrame(selected_samples_info)
        
        # 添加方向标签
        direction_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Unknown'}
        df['b2a_direction'] = df['label'].map(direction_map)
        
        # 调整列顺序
        cols_order = ['selection_order', 'original_index', 'camera_uuid', 'room', 
                     'frame_num_a', 'frame_num_b', 'b2a_direction', 'difficulty', 'augmentation_method']
        df = df[cols_order]
        
        return df
    
    def _print_statistics(self, selected_indices: List[int], all_difficulties: np.ndarray, final_matrix: np.ndarray, sample_info: Dict):
        """打印统计信息"""
        print(f"\n成功选择了 {len(selected_indices)} 个样本")
        
        # 难度分布统计
        if selected_indices:
            selected_difficulties = [all_difficulties[idx] for idx in selected_indices]
            difficulty_map_rev = {0: 'Easy', 1: 'Normal', 2: 'Hard'}
            difficulty_distribution = Counter([
                difficulty_map_rev[d] for d in selected_difficulties if d != -1
            ])
            print(f"难度分布: {dict(difficulty_distribution)}")
            
            # 增强方法分布统计
            selected_augmentations = [sample_info[idx]['augmentation_method'] for idx in selected_indices]
            augmentation_distribution = Counter(selected_augmentations)
            print(f"增强方法分布: {dict(augmentation_distribution)}")
        
        # 矩阵统计
        if final_matrix.size > 0:
            singular_values = np.linalg.svd(final_matrix, compute_uv=False)
            print(f"最终矩阵形状: {final_matrix.shape}")
            print(f"条件数: {singular_values[0] / singular_values[-1]:.2f}")
            print(f"最小奇异值: {singular_values[-1]:.6f}")


def main():

    # 定义模型路径：二维列表，第一维是模型，第二维是该模型下不同增强方法的CSV
    model_paths = [
        # RCF模型下的不同增强方法
        [
           "/home/zyz/Codes/SpatialCognition/Results/diverse_result/rcf_x2/different_methods/best_eval_model_test/rgb/none/softmax_rgb_none.csv",
           "/home/zyz/Codes/SpatialCognition/Results/diverse_result/rcf_x2/different_methods/best_eval_model_test/rgb/edge_reduction/softmax_rgb_edge_reduction.csv",
           "/home/zyz/Codes/SpatialCognition/Results/diverse_result/rcf_x2/different_methods/best_eval_model_test/rgb/texture_reduction/softmax_rgb_texture_reduction.csv",
           "/home/zyz/Codes/SpatialCognition/Results/diverse_result/rcf_x2/different_methods/best_eval_model_test/rgb/gray/softmax_rgb_gray.csv",
        ],
        # DeepTen模型下的不同增强方法
        [
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/deepten_x2/different_methods/best_eval_model_test/rgb/none/softmax_rgb_none.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/deepten_x2/different_methods/best_eval_model_test/rgb/edge_reduction/softmax_rgb_edge_reduction.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/deepten_x2/different_methods/best_eval_model_test/rgb/texture_reduction/softmax_rgb_texture_reduction.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/deepten_x2/different_methods/best_eval_model_test/rgb/gray/softmax_rgb_gray.csv",
        ], 
        # FC4 模型下的不同增强方法
        [
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/fc4_x2/different_methods/best_eval_model_test/rgb/none/softmax_rgb_none.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/fc4_x2/different_methods/best_eval_model_test/rgb/edge_reduction/softmax_rgb_edge_reduction.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/fc4_x2/different_methods/best_eval_model_test/rgb/texture_reduction/softmax_rgb_texture_reduction.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/fc4_x2/different_methods/best_eval_model_test/rgb/gray/softmax_rgb_gray.csv",
        ],
        # DPT模型下的不同增强方法
        [
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/dpt_x2/different_methods/best_eval_model_test/rgb/none/softmax_rgb_none.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/dpt_x2/different_methods/best_eval_model_test/rgb/edge_reduction/softmax_rgb_edge_reduction.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/dpt_x2/different_methods/best_eval_model_test/rgb/texture_reduction/softmax_rgb_texture_reduction.csv",
            "/home/zyz/Codes/SpatialCognition/Results/diverse_result/dpt_x2/different_methods/best_eval_model_test/rgb/gray/softmax_rgb_gray.csv",
        ]
    ]

    # 参数设置
    N = 30
    lambda_param = 1e-4
    label_filter = 3
    balance_difficulty = True
    balance_augmentation = True
    
    # 创建更严格的样本选择器
    selector = SampleSelector(
        lambda_param=lambda_param,
        strict_accuracy=True,  # 启用严格模式
        accuracy_tolerance=0.1  # 更严格的容忍度
    )
    
    # 输出路径设置
    output_path = 'ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance'
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # 创建样本选择器并运行
        selector = SampleSelector(lambda_param=lambda_param)
        selected_indices, final_matrix, selected_samples_df = selector.select_diverse_samples(
            model_paths, N, label_filter, balance_difficulty, balance_augmentation
        )
        
        # 保存结果
        if not selected_samples_df.empty:
            sample_info_csv = f'{output_path}/samples_{label_filter}.csv'
            selected_samples_df.to_csv(sample_info_csv, index=False)
            print(f"样本信息已保存到: {sample_info_csv}")
        
        if final_matrix.size > 0:
            matrix_csv = f'{output_path}/matrix_{label_filter}.csv'
            matrix_df = pd.DataFrame(final_matrix)
            
            # 生成列名
            matrix_cols = []
            for j in range(final_matrix.shape[1] // 4):
                for k in range(4):
                    matrix_cols.append(f'sample_{j}_softmax_{k}')
            
            if len(matrix_cols) == final_matrix.shape[1]:
                matrix_df.columns = matrix_cols
            
            matrix_df.to_csv(matrix_csv, index=False)
            print(f"结果矩阵已保存到: {matrix_csv}")
            
            # 保存特定标签的概率矩阵
            if label_filter is not None:
                prob_matrix_csv = f'{output_path}/matrix_{label_filter}_prob.csv'
                # 这里需要重新加载数据来构建概率矩阵
                model_dfs = selector.load_and_merge_model_data(model_paths)
                M = len(model_dfs)
                N_samples = len(selected_indices)
                prob_matrix = np.zeros((N_samples, M), dtype=np.float64)
                
                for m in range(M):
                    df = model_dfs[m]
                    for n, idx in enumerate(selected_indices):
                        prob_matrix[n, m] = df.loc[idx, f'softmax_{label_filter}']
                
                prob_df = pd.DataFrame(prob_matrix)
                prob_df.columns = [f'model_{i}' for i in range(M)]
                prob_df.index = [f'sample_{i}' for i in range(N_samples)]
                prob_df.to_csv(prob_matrix_csv)
                print(f"概率矩阵已保存到: {prob_matrix_csv}")
        
        print("\n样本选择完成！")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()