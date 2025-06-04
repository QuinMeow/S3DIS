'''
合并多个不同类别的样本到同一个csv文件中
'''
import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from tqdm import tqdm
import argparse

class SampleResultsCollector:
    """收集和汇总样本预测结果的工具类"""
    
    def __init__(self):
        self.model_names = ['RCF', 'DeepTen', 'FC4', 'DPT']
        self.augmentation_methods = ['none', 'edge_reduction', 'texture_reduction', 'gray']
        self.direction_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right', 4: 'Unknown'}
        
    def load_model_data(self, model_paths: List[List[str]]) -> Dict[Tuple[int, str], pd.DataFrame]:
        """
        加载所有模型的预测数据
        
        Args:
            model_paths: 二维列表，第一维是模型，第二维是该模型下不同增强方法的CSV路径
            
        Returns:
            字典，键为(model_index, augmentation_method)，值为对应的DataFrame
        """
        model_data = {}
        
        for model_idx, augmentation_paths in enumerate(model_paths):
            model_name = self.model_names[model_idx] if model_idx < len(self.model_names) else f"Model_{model_idx}"
            print(f"加载模型 {model_name} 的数据...")
            
            for aug_path in augmentation_paths:
                try:
                    df = pd.read_csv(aug_path)
                    # 从路径中提取增强方法名
                    aug_method = self._extract_augmentation_method(aug_path)
                    
                    # 添加模型标识
                    df['model_id'] = model_idx
                    df['model_name'] = model_name
                    df['augmentation_method'] = aug_method
                    
                    model_data[(model_idx, aug_method)] = df
                    print(f"  - 增强方法 {aug_method}: {len(df)} 条记录")
                    
                except Exception as e:
                    print(f"警告: 加载文件 {aug_path} 失败: {e}")
                    continue
        
        return model_data
    
    def _extract_augmentation_method(self, file_path: str) -> str:
        """从文件路径中提取增强方法名"""
        path_parts = file_path.split('/')
        
        # 查找包含增强方法的路径部分
        for part in reversed(path_parts):
            if part in self.augmentation_methods:
                return part
        
        # 如果没找到，从文件名中提取
        filename = os.path.basename(file_path)
        for method in self.augmentation_methods:
            if method in filename:
                return method
        
        return 'unknown'
    
    def load_sample_files(self, sample_file_paths: List[str]) -> pd.DataFrame:
        """
        加载所有的samples_i.csv文件并合并
        
        Args:
            sample_file_paths: samples_i.csv文件路径列表
            
        Returns:
            合并后的样本DataFrame
        """
        all_samples = []
        
        for file_path in sample_file_paths:
            try:
                df = pd.read_csv(file_path)
                # 从文件名中提取标签信息
                filename = os.path.basename(file_path)
                if 'samples_' in filename:
                    label_str = filename.replace('samples_', '').replace('.csv', '')
                    try:
                        label = int(label_str)
                        df['source_label'] = label
                        df['source_file'] = filename
                    except ValueError:
                        df['source_label'] = -1
                        df['source_file'] = filename
                
                all_samples.append(df)
                print(f"加载样本文件: {filename}, 样本数: {len(df)}")
                
            except Exception as e:
                print(f"警告: 加载样本文件 {file_path} 失败: {e}")
                continue
        
        if not all_samples:
            raise ValueError("没有成功加载任何样本文件")
        
        # 合并所有样本
        combined_samples = pd.concat(all_samples, ignore_index=True)
        print(f"\n总计加载样本数: {len(combined_samples)}")
        
        return combined_samples
    
    def match_sample_predictions(self, 
                               samples_df: pd.DataFrame, 
                               model_data: Dict[Tuple[int, str], pd.DataFrame]) -> pd.DataFrame:
        """
        根据样本信息匹配各模型的预测结果
        
        Args:
            samples_df: 样本信息DataFrame
            model_data: 模型数据字典
            
        Returns:
            包含预测结果的完整DataFrame
        """
        results = []
        
        print("开始匹配样本预测结果...")
        
        for idx, sample in tqdm(samples_df.iterrows(), total=len(samples_df), desc="处理样本"):
            sample_result = sample.to_dict()
            
            # 为每个模型收集预测结果
            for model_idx in range(len(self.model_names)):
                model_name = self.model_names[model_idx]
                aug_method = sample['augmentation_method']
                
                # 获取对应模型和增强方法的数据
                key = (model_idx, aug_method)
                if key not in model_data:
                    print(f"警告: 找不到模型 {model_name} 增强方法 {aug_method} 的数据")
                    continue
                
                model_df = model_data[key]
                
                # 使用标识字段进行匹配
                id_cols = ['camera_uuid', 'room', 'frame_num_a', 'frame_num_b']
                
                # 检查必要字段是否存在
                missing_fields = False
                for col in id_cols:
                    if col not in sample or col not in model_df.columns:
                        print(f"警告: 缺少匹配字段 {col}")
                        missing_fields = True
                        break
                
                if missing_fields:
                    continue
                
                # 构建匹配条件
                match_condition = (model_df['camera_uuid'] == sample['camera_uuid']) & \
                                (model_df['room'] == sample['room']) & \
                                (model_df['frame_num_a'] == sample['frame_num_a']) & \
                                (model_df['frame_num_b'] == sample['frame_num_b'])
                
                # 查找匹配的行
                matched_rows = model_df[match_condition]
                
                if len(matched_rows) == 0:
                    print(f"警告: 在模型 {model_name} 增强方法 {aug_method} 中找不到匹配的样本")
                    print(f"  查找条件: {[(col, sample[col]) for col in id_cols]}")
                    continue
                elif len(matched_rows) > 1:
                    print(f"警告: 在模型 {model_name} 增强方法 {aug_method} 中找到多个匹配的样本，使用第一个")
                
                # 获取预测结果
                model_row = matched_rows.iloc[0]
                
                # 添加预测相关信息
                softmax_cols = [f'softmax_{k}' for k in range(5)]  # 假设有5个类别
                for col in softmax_cols:
                    if col in model_row:
                        sample_result[f'{model_name}_{col}'] = model_row[col]
                
                # 添加预测标签
                if 'pred_label' in model_row:
                    sample_result[f'{model_name}_pred_label'] = model_row['pred_label']
                else:
                    # 从softmax中计算预测标签
                    softmax_values = [model_row.get(f'softmax_{k}', 0) for k in range(5)]
                    pred_label = np.argmax(softmax_values)
                    sample_result[f'{model_name}_pred_label'] = pred_label
                
                # 添加预测方向
                pred_label = sample_result[f'{model_name}_pred_label']
                sample_result[f'{model_name}_pred_direction'] = self.direction_map.get(pred_label, 'Unknown')
                
                # 添加是否预测正确
                true_label = sample.get('label', sample.get('source_label', -1))
                if true_label != -1:
                    is_correct = (pred_label == true_label)
                    sample_result[f'{model_name}_is_correct'] = is_correct
            
            results.append(sample_result)
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def generate_summary_statistics(self, results_df: pd.DataFrame) -> Dict:
        """生成汇总统计信息"""
        stats = {}
        
        # 基本统计
        stats['total_samples'] = len(results_df)
        stats['source_files'] = results_df['source_file'].unique().tolist()
        
        # 各标签分布
        if 'source_label' in results_df.columns:
            label_dist = results_df['source_label'].value_counts().to_dict()
            stats['label_distribution'] = label_dist
        
        # 难度分布
        if 'difficulty' in results_df.columns:
            difficulty_dist = results_df['difficulty'].value_counts().to_dict()
            stats['difficulty_distribution'] = difficulty_dist
        
        # 增强方法分布
        if 'augmentation_method' in results_df.columns:
            aug_dist = results_df['augmentation_method'].value_counts().to_dict()
            stats['augmentation_distribution'] = aug_dist
        
        # 各模型准确率
        model_accuracy = {}
        for model_name in self.model_names:
            correct_col = f'{model_name}_is_correct'
            if correct_col in results_df.columns:
                accuracy = results_df[correct_col].mean()
                model_accuracy[model_name] = accuracy
        stats['model_accuracy'] = model_accuracy
        
        # 模型一致性统计
        pred_cols = [f'{model_name}_pred_label' for model_name in self.model_names]
        available_pred_cols = [col for col in pred_cols if col in results_df.columns]
        
        if len(available_pred_cols) > 1:
            # 计算模型间的一致性
            consistency_scores = []
            for idx, row in results_df.iterrows():
                predictions = [row[col] for col in available_pred_cols]
                # 计算最常见预测的比例
                from collections import Counter
                pred_counts = Counter(predictions)
                most_common_count = pred_counts.most_common(1)[0][1]
                consistency = most_common_count / len(predictions)
                consistency_scores.append(consistency)
            
            stats['average_model_consistency'] = np.mean(consistency_scores)
            stats['model_agreement_distribution'] = pd.Series(consistency_scores).value_counts().to_dict()
        
        return stats
    
    def save_results(self, results_df: pd.DataFrame, output_path: str, 
                    include_stats: bool = True) -> None:
        """
        保存结果到文件
        
        Args:
            results_df: 结果DataFrame
            output_path: 输出文件路径
            include_stats: 是否包含统计信息
        """
        # 保存主要结果
        results_df.to_csv(output_path, index=False)
        print(f"结果已保存到: {output_path}")
        
        if include_stats:
            # 生成并保存统计信息
            stats = self.generate_summary_statistics(results_df)
            
            # 保存统计信息到文本文件
            stats_path = output_path.replace('.csv', '_statistics.txt')
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("样本预测结果统计报告\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in stats.items():
                    f.write(f"{key}:\n")
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"  {value}\n")
                    f.write("\n")
            
            print(f"统计信息已保存到: {stats_path}")
            
            # 打印简要统计
            print(f"\n简要统计:")
            print(f"总样本数: {stats['total_samples']}")
            if 'model_accuracy' in stats:
                print("各模型准确率:")
                for model, acc in stats['model_accuracy'].items():
                    print(f"  {model}: {acc:.4f}")
    
    def process_samples(self, 
                       model_paths: List[List[str]], 
                       sample_file_paths: List[str], 
                       output_path: str) -> pd.DataFrame:
        """
        主处理函数
        
        Args:
            model_paths: 模型路径配置
            sample_file_paths: 样本文件路径列表
            output_path: 输出文件路径
            
        Returns:
            处理后的结果DataFrame
        """
        print("开始处理样本预测结果...")
        
        # 1. 加载模型数据
        model_data = self.load_model_data(model_paths)
        
        # 2. 加载样本文件
        samples_df = self.load_sample_files(sample_file_paths)
        
        # 3. 匹配预测结果
        results_df = self.match_sample_predictions(samples_df, model_data)
        
        # 4. 保存结果
        self.save_results(results_df, output_path)
        
        return results_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='收集和汇总样本预测结果')
    parser.add_argument('--sample_files', nargs='+', required=True,
                       help='samples_i.csv文件路径列表')
    parser.add_argument('--output', '-o', required=True,
                       help='输出文件路径')
    parser.add_argument('--output_dir', default='.',
                       help='输出目录 (默认当前目录)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output)
    
    # 定义模型路径配置 (与findDiverseSamples.py中一致)
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
    
    try:
        # 创建处理器并运行
        collector = SampleResultsCollector()
        results_df = collector.process_samples(
            model_paths=model_paths,
            sample_file_paths=args.sample_files,
            output_path=output_path
        )
        
        print(f"\n处理完成! 共处理 {len(results_df)} 个样本")
        print(f"结果文件: {output_path}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 如果直接运行，可以使用示例配置
    if len(os.sys.argv) == 1:
        print("示例用法:")
        print("python getDiveseSamplesResults.py --sample_files samples_0.csv samples_1.csv --output combined_results.csv")
        print("\n或者使用默认示例:")
        
        # 示例配置
        sample_files = [
            "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance/samples_0.csv",
            "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance/samples_1.csv",
            "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance/samples_2.csv",
            "/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance/samples_3.csv"
        ]
        output_file = "ImagePairsFromPano/5classes_dataset/diverse_sample_Enhance/combined_sample_results.csv"
        
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
        
        collector = SampleResultsCollector()
        results_df = collector.process_samples(
            model_paths=model_paths,
            sample_file_paths=sample_files,
            output_path=output_file
        )
        
        print(f"\n示例运行完成! 处理了 {len(results_df)} 个样本")
    else:
        main()