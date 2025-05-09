import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter
import numba # 导入 numba
from tqdm import tqdm # 导入 tqdm

def load_model_outputs(file_paths: List[str]) -> List[pd.DataFrame]:
    """加载多个模型的softmax输出文件"""
    model_outputs = []
    for path in file_paths:
        df = pd.read_csv(path)
        model_outputs.append(df)
    return model_outputs

# 使用 Numba 加速矩阵构建
@numba.jit(nopython=True)
def build_matrix_numba(all_softmax_data: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    """
    为选定的样本构建模型预测矩阵 (Numba 加速版)
    参数:
    - all_softmax_data: 形状为 (M, total_samples, 4) 的 NumPy 数组，包含所有模型的softmax值
    - sample_indices: 包含所选样本索引的 NumPy 数组
    """
    M, _, _ = all_softmax_data.shape
    N = len(sample_indices)
    
    # 初始化矩阵 (M x 4N)
    matrix = np.zeros((M, 4*N), dtype=np.float64) # 指定 dtype
    
    for i in range(M): # 遍历模型
        for j in range(N): # 遍历选中的样本索引
            idx = sample_indices[j]
            # 从预处理数据中提取 softmax 值
            softmax_values = all_softmax_data[i, idx, :]
            # 放入矩阵
            matrix[i, j*4:(j+1)*4] = softmax_values
            
    return matrix

def filter_samples_by_criteria(model_df: pd.DataFrame, label_filter: Optional[int] = None) -> List[int]:
    """根据标签筛选样本"""
    if label_filter is not None:
        return model_df[model_df['label'] == label_filter].index.tolist()
    return list(range(len(model_df)))

def select_diverse_samples(
    all_softmax_data: np.ndarray, 
    all_difficulties: np.ndarray, 
    initial_indices: List[int], 
    N: int, 
    lambda_param: float = 0.1, 
    balance_difficulty: bool = False
) -> List[int]:
    """
    选择在模型间预测差异最大的N个样本 (使用 Numba 和 tqdm)
    
    参数:
    - all_softmax_data: 形状为 (M, total_samples, 4) 的 NumPy 数组
    - all_difficulties: 形状为 (total_samples,) 的 NumPy 数组 (0: Easy, 1: Normal, 2: Hard, -1: Unknown)
    - initial_indices: 经过标签过滤后的初始样本索引列表
    - N: 要选择的样本数量
    - lambda_param: 平衡参数
    - balance_difficulty: 是否平衡不同难度等级的样本
    """
    
    if len(initial_indices) < N:
        raise ValueError(f"可用样本数量({len(initial_indices)})小于请求数量({N})")
    
    # 初始化
    selected_indices_list = []
    remaining_indices_set = set(initial_indices) # 使用集合以提高移除效率
    
    # 如果需要平衡难度等级
    if balance_difficulty:
        # 排除Unknown难度 (-1)
        valid_mask = all_difficulties[initial_indices] != -1
        valid_initial_indices = np.array(initial_indices)[valid_mask]
        remaining_indices_set = set(valid_initial_indices)
        
        if len(remaining_indices_set) < N:
             raise ValueError(f"排除Unknown难度后，可用样本数量({len(remaining_indices_set)})小于请求数量({N})")

        # 计算每个难度级别的目标数量
        target_counts = {0: N//3, 1: N//3, 2: N - 2*(N//3)} # 0: Easy, 1: Normal, 2: Hard
        current_counts = {0: 0, 1: 0, 2: 0}
        
        # 预先按难度分组索引，提高查找效率
        difficulty_map = {0: [], 1: [], 2: []}
        for idx in remaining_indices_set:
            difficulty_code = all_difficulties[idx]
            if difficulty_code != -1:
                 difficulty_map[difficulty_code].append(idx)
        
        # 使用 tqdm 添加进度条
        pbar = tqdm(range(N), desc="Selecting diverse samples (balanced)")
        for _ in pbar:
            best_gain = float("-inf")
            best_idx = -1 # 使用-1表示未找到
            
            # 确定当前最需要增加的难度级别
            candidate_indices_pool = []
            if len(selected_indices_list) < N - 3: # 留出最后几个样本用于微调平衡
                # 计算各难度与目标的差距比例
                needed_ratios = {}
                total_selected = max(1, len(selected_indices_list)) # 避免除以零
                for d_code in target_counts:
                    target_ratio = target_counts[d_code] / N
                    current_ratio = current_counts[d_code] / total_selected if total_selected > 0 else 0
                    needed_ratios[d_code] = target_ratio - current_ratio
                
                # 按需求度排序
                preferred_difficulties = sorted(needed_ratios, key=needed_ratios.get, reverse=True)
                
                # 按优先级构建候选池
                for d_code in preferred_difficulties:
                    # 只考虑仍在 remaining_indices_set 中的样本
                    potential_candidates = [idx for idx in difficulty_map[d_code] if idx in remaining_indices_set]
                    if potential_candidates:
                        candidate_indices_pool = potential_candidates
                        break # 找到一个非空候选集就跳出
                
                if not candidate_indices_pool: # 如果按优先级没找到，则考虑所有剩余样本
                    candidate_indices_pool = list(remaining_indices_set)

            else: # 最后几个样本，优先补足未达标的难度
                missing_counts = {d_code: max(0, target_counts[d_code] - current_counts[d_code]) for d_code in target_counts}
                if sum(missing_counts.values()) > 0:
                    # 按缺失数量排序
                    missing_difficulties = sorted(missing_counts, key=missing_counts.get, reverse=True)
                    for d_code in missing_difficulties:
                        if missing_counts[d_code] > 0:
                            potential_candidates = [idx for idx in difficulty_map[d_code] if idx in remaining_indices_set]
                            if potential_candidates:
                                candidate_indices_pool = potential_candidates
                                break
                    if not candidate_indices_pool:
                         candidate_indices_pool = list(remaining_indices_set)
                else: # 所有难度都达标或超标，则考虑所有剩余样本
                    candidate_indices_pool = list(remaining_indices_set)

            # 从候选样本中选择最优的
            current_selection_np = np.array(selected_indices_list, dtype=np.int64) # Numba 需要 NumPy 数组
            for idx in candidate_indices_pool:
                # 尝试添加样本
                S_try_np = np.append(current_selection_np, idx)
                
                # 使用 Numba 加速的 build_matrix
                A_try = build_matrix_numba(all_softmax_data, S_try_np)
                
                # 计算奇异值
                try:
                    s = np.linalg.svd(A_try, compute_uv=False)  # 奇异值降序排列
                    # if len(s) < len(S_try_np): # 检查奇异值数量是否足够
                        # print(f"Warning: Rank deficiency detected for sample set size {len(S_try_np)}. Skipping index {idx}.")
                        # continue # 跳过可能导致问题的样本
                    sigma_min = s[-1]
                    sigma_max = s[0]
                    
                    # 防止除以零或非常小的值
                    if sigma_min < 1e-10: 
                        J = -float('inf') # 惩罚数值不稳定的情况
                    else:
                        # 目标函数
                        J = sigma_min - lambda_param * (sigma_max / sigma_min - 1)

                    if J > best_gain:
                        best_gain = J
                        best_idx = idx
                except np.linalg.LinAlgError:
                    print(f"Warning: SVD computation failed for index {idx}. Skipping.")
                    continue # SVD计算失败则跳过

            if best_idx != -1:
                selected_indices_list.append(best_idx)
                remaining_indices_set.remove(best_idx)
                # 更新当前各难度级别的计数
                difficulty_code = all_difficulties[best_idx]
                current_counts[difficulty_code] += 1
                # 更新进度条显示信息
                pbar.set_postfix(difficulty_counts=current_counts)
            else:
                print("\nWarning: Could not find a suitable sample to add. Stopping early.")
                break # 无法找到更好的样本
        pbar.close()

    else: # 原始贪心选择逻辑(不考虑难度平衡)
        # 使用 tqdm 添加进度条
        pbar = tqdm(range(N), desc="Selecting diverse samples")
        for _ in pbar:
            best_gain = float("-inf")
            best_idx = -1

            current_selection_np = np.array(selected_indices_list, dtype=np.int64)
            # 将 set 转换为 list 进行迭代
            candidate_indices_pool = list(remaining_indices_set) 
            
            for idx in candidate_indices_pool:
                S_try_np = np.append(current_selection_np, idx)
                A_try = build_matrix_numba(all_softmax_data, S_try_np)
                
                try:
                    s = np.linalg.svd(A_try, compute_uv=False)
                    if len(s) < len(S_try_np):
                        continue
                    sigma_min = s[-1]
                    sigma_max = s[0]

                    if sigma_min < 1e-10:
                        J = -float('inf')
                    else:
                        J = sigma_min - lambda_param * (sigma_max / sigma_min - 1)
                    
                    if J > best_gain:
                        best_gain = J
                        best_idx = idx
                except np.linalg.LinAlgError:
                    continue

            if best_idx != -1:
                selected_indices_list.append(best_idx)
                remaining_indices_set.remove(best_idx)
            else:
                print("\nWarning: Could not find a suitable sample to add. Stopping early.")
                break
        pbar.close()
    
    # 统计所选样本的难度分布
    if balance_difficulty:
        final_difficulties = [all_difficulties[idx] for idx in selected_indices_list]
        # 将数字代码映射回字符串
        difficulty_map_rev = {0: 'Easy', 1: 'Normal', 2: 'Hard'}
        difficulty_distribution = Counter([difficulty_map_rev[d] for d in final_difficulties])
        print(f"最终难度分布: {dict(difficulty_distribution)}")
    
    return selected_indices_list

def verify_sample_consistency(model_dfs: List[pd.DataFrame], selected_indices: List[int]) -> bool:
    """验证选定的样本在所有模型中对应相同的数据点"""
    if not selected_indices: # 如果没有选出样本，则无需验证
        return True
        
    id_cols = ['camera_uuid', 'room', 'frame_num_a', 'frame_num_b']
    
    # 使用第一个选定样本的第一个模型作为参考
    reference_idx = selected_indices[0]
    reference_values = model_dfs[0].iloc[reference_idx][id_cols].values
    
    for idx in selected_indices:
        for df in model_dfs:
            current_values = df.iloc[idx][id_cols].values
            # 检查类型和值是否都相等
            if not all(ref == cur for ref, cur in zip(reference_values, current_values)):
                 print(f"Inconsistency found at index {idx}:")
                 print(f"  Reference ({model_dfs[0].iloc[reference_idx]['camera_uuid']}): {reference_values}")
                 print(f"  Current ({df.iloc[idx]['camera_uuid']}): {current_values}")
                 return False
            # 更新参考值为当前样本，以便后续比较基于同一行的不同模型
            reference_values = current_values 
        # 重置参考值为下一个样本的第一个模型
        if selected_indices.index(idx) + 1 < len(selected_indices):
             next_idx = selected_indices[selected_indices.index(idx) + 1]
             reference_values = model_dfs[0].iloc[next_idx][id_cols].values

    return True

def _calculate_balance_cost(is_correct_matrix: np.ndarray, selected_indices: List[int], w_accuracy_std: float, w_all_correct: float, w_all_incorrect: float) -> float:
    """
    计算平衡性成本：
    - 各模型正确率的标准差
    - 所有模型全对样本比例
    - 所有模型全错样本比例
    """
    if not selected_indices:
        return float('inf')
    sub_matrix = is_correct_matrix[:, selected_indices]  # (M, N')
    M, Np = sub_matrix.shape
    # 各模型正确率
    accs = sub_matrix.sum(axis=1) / Np
    acc_std = np.std(accs)
    # 每个样本被所有模型全对/全错
    all_correct = np.all(sub_matrix, axis=0).sum() / Np
    all_incorrect = np.all(~sub_matrix, axis=0).sum() / Np
    cost = w_accuracy_std * acc_std + w_all_correct * all_correct + w_all_incorrect * all_incorrect
    return cost

def filter_for_balanced_samples(
    is_correct_matrix: np.ndarray,
    candidate_indices: List[int],
    N_prime: int,

    all_difficulties: np.ndarray,
    balance_difficulty: bool = False,
    w_accuracy_std: float = 1.0,
    w_all_correct: float = 1.0,
    w_all_incorrect: float = 1.0
) -> List[int]:
    """
    从候选样本中筛选N'个平衡性更好的子集，支持难度平衡
    """
    best_indices = []
    remaining = set(candidate_indices)
    if balance_difficulty:
        # 统计候选池中各难度的样本
        candidate_difficulties = {idx: all_difficulties[idx] for idx in candidate_indices}
        valid_difficulties = [d for d in [0,1,2] if any(v==d for v in candidate_difficulties.values())]
        target_counts = {d: N_prime // len(valid_difficulties) for d in valid_difficulties}
        for i, d in enumerate(valid_difficulties[:N_prime % len(valid_difficulties)]):
            target_counts[d] += 1
        current_counts = {d: 0 for d in valid_difficulties}
    for _ in range(N_prime):
        best_cost = float('inf')
        best_idx = None
        candidate_pool = list(remaining)
        if balance_difficulty:
            # 计算每个难度还缺多少
            needed = {d: target_counts[d] - current_counts[d] for d in target_counts}
            # 只考虑还没补满的
            lacking = [d for d, v in needed.items() if v > 0]
            if lacking:
                # 选最缺的
                max_d = max(lacking, key=lambda d: needed[d])
                candidate_pool = [idx for idx in remaining if all_difficulties[idx] == max_d]
            else:
                candidate_pool = list(remaining)
        for idx in candidate_pool:
            trial = best_indices + [idx]
            cost = _calculate_balance_cost(is_correct_matrix, trial, w_accuracy_std, w_all_correct, w_all_incorrect)
            if cost < best_cost:
                best_cost = cost
                best_idx = idx
        if best_idx is not None:
            best_indices.append(best_idx)
            remaining.remove(best_idx)
            if balance_difficulty:
                d = all_difficulties[best_idx]
                if d in current_counts:
                    current_counts[d] += 1
        else:
            break
    return best_indices

def find_diverse_samples(
    file_paths: List[str], N: int, lambda_param: float = 0.1,
    label_filter: Optional[int] = None, balance_difficulty: bool = False,
    N_prime: Optional[int] = None, w_accuracy_std: float = 1.0, w_all_correct: float = 1.0, w_all_incorrect: float = 1.0
) -> Tuple[List[int], np.ndarray, pd.DataFrame]:
    """
    主函数：运行差异样本选择算法 (使用 Numba 和 tqdm)
    返回:
    - selected_indices: 选定样本的索引列表
    - final_matrix: 最终构建的 M x 4N 矩阵
    - selected_samples_df: 包含所选样本详细信息的 DataFrame
    """
    # 加载模型输出
    print("Loading model outputs...")
    model_dfs = load_model_outputs(file_paths)
    
    # --- 数据预处理 ---
    print("Preprocessing data...")
    M = len(model_dfs)
    total_samples = len(model_dfs[0])
    all_softmax_data = np.zeros((M, total_samples, 4), dtype=np.float64)
    
    # 提取所有模型的 softmax 数据
    for i, df in enumerate(model_dfs):
        softmax_cols = [f'softmax_{k}' for k in range(4)]
        all_softmax_data[i, :, :] = df[softmax_cols].values
        
    # 提取难度信息并编码 (只需从第一个 DataFrame 获取)
    difficulty_map_encode = {'Easy': 0, 'Normal': 1, 'Hard': 2, 'Unknown': -1}
    # 使用 .get 提供默认值 -1 处理可能的缺失或非标准难度标签
    all_difficulties = np.array([difficulty_map_encode.get(d, -1) for d in model_dfs[0]['difficulty']], dtype=np.int64)

    # 按标签过滤初始索引
    initial_indices = filter_samples_by_criteria(model_dfs[0], label_filter)
    print(f"Initial samples after label filtering ({label_filter=}): {len(initial_indices)}")

    # --- 选择差异样本 ---
    selected_indices = select_diverse_samples(
        all_softmax_data, 
        all_difficulties, 
        initial_indices, 
        N, 
        lambda_param, 
        balance_difficulty
    )

    # --- 平衡性筛选 ---
    if label_filter is not None and N_prime is not None and N_prime < len(selected_indices):
        # 构建 is_correct_matrix: (M, total_samples)
        M = len(model_dfs)
        total_samples = len(model_dfs[0])
        # 每个模型对每个样本的预测类别
        pred_labels = np.argmax(all_softmax_data, axis=2)  # (M, total_samples)
        is_correct_matrix = (pred_labels == label_filter)  # (M, total_samples)
        # 只对已选的N个样本做二次筛选
        balanced_indices = filter_for_balanced_samples(
            is_correct_matrix,
            selected_indices,
            N_prime,
            all_difficulties,
            balance_difficulty,
            w_accuracy_std,
            w_all_correct,
            w_all_incorrect
        )
        selected_indices = balanced_indices
        print(f"\n经过平衡性筛选后，最终选定的样本索引: {selected_indices}")

    # --- 验证与收尾 ---
    print("Verifying sample consistency...")
    is_consistent = verify_sample_consistency(model_dfs, selected_indices)
    if not is_consistent:
        raise ValueError("所选样本在不同模型间不一致 (camera_uuid, room, frame_num_a, frame_num_b 必须匹配)")
    
    print("Building final matrix...")
    # 构建最终矩阵 (使用 Numba 加速版)
    final_matrix = build_matrix_numba(all_softmax_data, np.array(selected_indices, dtype=np.int64))
    
    # --- 收集并输出所选样本的信息 ---
    selected_samples_info = []
    print("\n所选样本的详细信息:")
    if selected_indices:
        info_cols = ['camera_uuid', 'room', 'frame_num_a', 'frame_num_b', 'label', 'difficulty']
        for i, idx in enumerate(selected_indices):
            # 从原始 DataFrame 获取信息以显示
            sample_data = model_dfs[0].iloc[idx][info_cols].to_dict()
            sample_data['original_index'] = idx # 添加原始索引
            selected_samples_info.append(sample_data)
            print(f"{i+1}. 索引 {idx}: {sample_data}")
        
        # 创建 DataFrame
        selected_samples_df = pd.DataFrame(selected_samples_info)
        # 调整列顺序，将 original_index 放在前面
        cols_order = ['original_index'] + info_cols
        selected_samples_df = selected_samples_df[cols_order]

    else:
        print("未能选出任何样本。")
        selected_samples_df = pd.DataFrame() # 返回空 DataFrame

    return selected_indices, final_matrix, selected_samples_df # 返回 DataFrame

if __name__ == "__main__":
    # 示例用法
    file_paths = [
        '/home/zyz/Codes/SpatialCognition/Results/x2_result/rcf_x2_swin_base_patch4_window7_224_20250424-2140/best_eval_model_test/none/softmax_test.csv',
        '/home/zyz/Codes/SpatialCognition/Results/x2_result/deepten_x2_swin_base_patch4_window7_224_20250504-1116/best_eval_model_test/none/softmax_test.csv',
        '/home/zyz/Codes/SpatialCognition/Results/x2_result/dpt_dept_x2_swin_base_patch4_window7_224_20250504-1556/best_eval_model_test/none/softmax_test.csv' 
    ]  # 模型输出文件列表
    
    N = 60  # 要选择的样本数
    lambda_param = 1e-4  # 平衡参数
    label_filter = 1  # 可选：只选择标签为 0 的样本，设为None表示选择所有标签
    balance_difficulty = True  # 开启难度平衡
    # 新增平衡性筛选参数
    N_prime = 30  # 最终平衡样本数
    w_accuracy_std = 1.0
    w_all_correct = 2.0
    w_all_incorrect = 2.0
    
    # 定义输出文件名
    output_path = 'ImagePairsFromPano/5classes_dataset/diverse_sample_csv'
    output_sample_info_csv = f'{output_path}/samples_{label_filter}.csv'
    output_matrix_csv = f'{output_path}/matrix_{label_filter}.csv'
    # output_sample_info_csv = '/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/selected_samples_info.csv'
    # output_matrix_csv = '/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/final_result_matrix.csv'

    try:
        selected_indices, final_matrix, selected_samples_df = find_diverse_samples( # 接收 DataFrame
            file_paths, N, lambda_param, label_filter, balance_difficulty,
            N_prime, w_accuracy_std, w_all_correct, w_all_incorrect
        )
        
        print(f"\n选定的样本索引: {selected_indices}")
        print(f"最终矩阵形状: {final_matrix.shape}")  # 应为 (M, 4N)
        
        # 计算最终矩阵的奇异值
        if final_matrix.size > 0: # 确保矩阵非空
             singular_values = np.linalg.svd(final_matrix, compute_uv=False)
             print(f"奇异值: {singular_values}")
             if singular_values[-1] > 1e-10: # 避免除以零
                 print(f"条件数: {singular_values[0] / singular_values[-1]}")
             else:
                 print("条件数: Inf (最小奇异值接近零)")
        else:
             print("最终矩阵为空，无法计算奇异值。")

        # --- 保存结果到 CSV ---
        if not selected_samples_df.empty:
            # 将label列替换为b2a_direction列，并将数字转为字符串
            direction_map_rev = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Unknown'}
            # 新增b2a_direction列
            selected_samples_df['b2a_direction'] = selected_samples_df['label'].map(direction_map_rev)
            # 调整列顺序，去掉label列
            cols_order = ['original_index', 'camera_uuid', 'room', 'frame_num_a', 'frame_num_b', 'b2a_direction', 'difficulty']
            selected_samples_df = selected_samples_df[cols_order]
            print(f"\nSaving selected sample info to {output_sample_info_csv}...")
            selected_samples_df.to_csv(output_sample_info_csv, index=False)
            print("Done.")
        
        if final_matrix.size > 0:
             print(f"Saving final result matrix to {output_matrix_csv}...")
             # 将 NumPy 矩阵转换为 DataFrame 以便保存带标题的 CSV
             matrix_df = pd.DataFrame(final_matrix)
             # 列名按最终样本数生成
             matrix_cols = []
             for i in range(final_matrix.shape[0]):
                 for j in range(final_matrix.shape[1] // (4 * final_matrix.shape[0])):
                     for k in range(4):
                         matrix_cols.append(f'model_{i}_sample_{j}_softmax_{k}')
             if len(matrix_cols) == final_matrix.shape[1]:
                 matrix_df.columns = matrix_cols
             else:
                 print(f"Warning: Column name count ({len(matrix_cols)}) does not match matrix columns ({final_matrix.shape[1]}). Saving without header.")

             matrix_df.to_csv(output_matrix_csv, index=False) # 保存矩阵
             print("Done.")
        else:
             print("Final matrix is empty, skipping saving.")

    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")