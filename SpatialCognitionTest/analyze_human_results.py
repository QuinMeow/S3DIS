import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import pandas as pd

def read_test_results(csv_file):
    """
    读取人类空间关系测试的结果CSV文件
    
    Args:
        csv_file: 结果CSV文件路径
    
    Returns:
        dataframe: 包含结果数据的DataFrame
    """
    try:
        # 尝试读取CSV文件
        df = pd.read_csv(csv_file, skipinitialspace=True)
        print(f"成功读取测试结果: {csv_file}")
        print(f"总测试样本数: {len(df)}")
        return df
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return None

def compute_metrics(cm, class_names):
    """
    基于混淆矩阵计算各类别的精确度、召回率和F1分数
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称列表
    
    Returns:
        metrics: 包含每个类别和总体的各项指标的字典
    """
    num_classes = cm.shape[0]
    metrics = {
        'class_metrics': [],
        'overall': {}
    }
    
    # 计算每个类别的指标
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_samples = np.sum(cm)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['class_metrics'].append({
            'class_name': class_names[i],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        })
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
    
    # 计算总体指标
    total_accuracy = np.trace(cm) / total_samples if total_samples > 0 else 0
    
    # 计算宏平均指标
    macro_precision = np.mean([m['precision'] for m in metrics['class_metrics']])
    macro_recall = np.mean([m['recall'] for m in metrics['class_metrics']])
    macro_f1 = np.mean([m['f1'] for m in metrics['class_metrics']])
    
    # 计算微平均指标
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    metrics['overall'] = {
        'accuracy': total_accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
    }
    
    return metrics

def analyze_by_difficulty(df, class_names, output_dir=None):
    """
    按难度级别分析测试结果
    
    Args:
        df: 包含测试结果的DataFrame
        class_names: 类别名称列表
        output_dir: 输出目录
        
    Returns:
        results: 包含按难度分组的评估结果的字典
    """
    # 转换为numpy数组方便操作
    all_labels = np.array([class_names.index(label) if label in class_names else -1 for label in df['b2a_direction']])
    all_preds = np.array([class_names.index(pred) if pred in class_names else -1 for pred in df['user_answer']])
    all_difficulties = np.array(df['difficulty'])
    
    # 排除无效标签
    valid_indices = (all_labels != -1) & (all_preds != -1)
    all_labels = all_labels[valid_indices]
    all_preds = all_preds[valid_indices]
    all_difficulties = all_difficulties[valid_indices]
    
    # 确保输出目录存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有难度级别
    unique_difficulties = np.unique([d for d in all_difficulties if d != "Unknown"])
    
    results = {'overall': {}, 'by_difficulty': {}}
    
    # 计算总体指标
    overall_cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    overall_accuracy = accuracy_score(all_labels, all_preds)
    overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', labels=range(len(class_names)))
    
    results['overall'] = {
        'confusion_matrix': overall_cm,
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'sample_count': len(all_labels)
    }
    
    # 计算每个难度级别的指标
    for difficulty in unique_difficulties:
        mask = (all_difficulties == difficulty)
        diff_preds = all_preds[mask]
        diff_labels = all_labels[mask]
        
        if len(diff_labels) == 0:
            continue
            
        diff_cm = confusion_matrix(diff_labels, diff_preds, labels=range(len(class_names)))
        diff_accuracy = accuracy_score(diff_labels, diff_preds)
        diff_precision, diff_recall, diff_f1, _ = precision_recall_fscore_support(
            diff_labels, diff_preds, average='macro', labels=range(len(class_names)))
            
        results['by_difficulty'][difficulty] = {
            'confusion_matrix': diff_cm,
            'accuracy': diff_accuracy,
            'precision': diff_precision,
            'recall': diff_recall,
            'f1': diff_f1,
            'sample_count': len(diff_labels)
        }
    
    # 计算每个方向的统计数据
    direction_stats = {}
    for i, direction in enumerate(class_names):
        mask = (all_labels == i)
        correct = np.sum((all_labels == i) & (all_preds == i))
        total = np.sum(mask)
        
        direction_stats[direction] = {
            'accuracy': correct / total if total > 0 else 0,
            'sample_count': int(total)
        }
    
    results['by_direction'] = direction_stats
    
    # 输出指标
    print_and_save_results(results, class_names, output_dir)
    
    return results

def print_and_save_results(results, class_names, output_dir):
    """
    打印并保存评估结果
    
    Args:
        results: 评估结果字典
        class_names: 类别名称列表
        output_dir: 输出目录
    """
    font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14)
    
    # 打印总体结果
    print("\n===== 总体评估结果 =====")
    print(f"样本数: {results['overall']['sample_count']}")
    print(f"准确率: {results['overall']['accuracy']:.4f}")
    print(f"精确率: {results['overall']['precision']:.4f}")
    print(f"召回率: {results['overall']['recall']:.4f}")
    print(f"F1分数: {results['overall']['f1']:.4f}")
    
    # 打印每个方向的结果
    print("\n===== 按方向评估结果 =====")
    for direction, stats in results['by_direction'].items():
        print(f"{direction}: 准确率 {stats['accuracy']:.4f} (样本数: {stats['sample_count']})")
    
    # 可视化总体混淆矩阵
    if output_dir:
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['overall']['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('人类判断 - 总体混淆矩阵', fontproperties=font)
        plt.ylabel('真实标签', fontproperties=font)
        plt.xlabel('预测标签', fontproperties=font)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_overall.png'))
        plt.close()
    
    # 保存总体评估结果到CSV
    if output_dir:
        with open(os.path.join(output_dir, 'metrics_overall.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['指标', '值'])
            writer.writerow(['样本数', results['overall']['sample_count']])
            writer.writerow(['准确率', f"{results['overall']['accuracy']:.4f}"])
            writer.writerow(['精确率', f"{results['overall']['precision']:.4f}"])
            writer.writerow(['召回率', f"{results['overall']['recall']:.4f}"])
            writer.writerow(['F1分数', f"{results['overall']['f1']:.4f}"])
    
    # 保存每个方向的结果到CSV
    if output_dir:
        with open(os.path.join(output_dir, 'metrics_by_direction.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['方向', '准确率', '样本数'])
            for direction, stats in results['by_direction'].items():
                writer.writerow([direction, f"{stats['accuracy']:.4f}", stats['sample_count']])
    
    # 打印并保存每个难度级别的结果
    print("\n===== 按难度级别的评估结果 =====")
    for difficulty, metrics in results['by_difficulty'].items():
        print(f"\n----- {difficulty} 难度 -----")
        print(f"样本数: {metrics['sample_count']}")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        
        # 可视化难度级别混淆矩阵
        if output_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'人类判断 - 混淆矩阵 ({difficulty}难度)', fontproperties=font)
            plt.ylabel('真实标签', fontproperties=font)
            plt.xlabel('预测标签', fontproperties=font)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{difficulty}.png'))
            plt.close()
        
        # 保存难度级别评估结果到CSV
        if output_dir:
            with open(os.path.join(output_dir, f'metrics_{difficulty}.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['指标', '值'])
                writer.writerow(['样本数', metrics['sample_count']])
                writer.writerow(['准确率', f"{metrics['accuracy']:.4f}"])
                writer.writerow(['精确率', f"{metrics['precision']:.4f}"])
                writer.writerow(['召回率', f"{metrics['recall']:.4f}"])
                writer.writerow(['F1分数', f"{metrics['f1']:.4f}"])

def analyze_response_times(df, output_dir=None):
    """
    分析响应时间分布
    
    Args:
        df: 包含测试结果的DataFrame
        output_dir: 输出目录
    """
    # 排除timeout结果
    valid_df = df[df['response_time'] > 0]
    
    if len(valid_df) == 0:
        print("没有有效的响应时间数据")
        return
    
    # 计算统计量
    mean_time = valid_df['response_time'].mean()
    median_time = valid_df['response_time'].median()
    std_time = valid_df['response_time'].std()
    
    print("\n===== 响应时间统计 =====")
    print(f"样本数: {len(valid_df)}")
    print(f"平均响应时间: {mean_time:.2f} 秒")
    print(f"中位响应时间: {median_time:.2f} 秒")
    print(f"响应时间标准差: {std_time:.2f} 秒")
    
    # 可视化响应时间分布
    if output_dir:
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_df['response_time'], kde=True, bins=20)
        plt.axvline(mean_time, color='r', linestyle='--', label=f'平均值: {mean_time:.2f}s')
        plt.axvline(median_time, color='g', linestyle='--', label=f'中位数: {median_time:.2f}s')
        plt.title('人类判断 - 响应时间分布', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14))
        plt.xlabel('响应时间 (秒)', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.ylabel('频次', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.legend(prop=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'response_times.png'))
        plt.close()
        
        # 按难度分析响应时间
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='difficulty', y='response_time', data=valid_df)
        plt.title('人类判断 - 不同难度的响应时间', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14))
        plt.xlabel('难度', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.ylabel('响应时间 (秒)', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'response_times_by_difficulty.png'))
        plt.close()
        
        # 按正确性分析响应时间
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='is_correct', y='response_time', data=valid_df)
        plt.title('人类判断 - 正确与错误答案的响应时间', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=14))
        plt.xlabel('是否正确', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.ylabel('响应时间 (秒)', fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.xticks([0, 1], ['错误', '正确'], fontproperties=FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'response_times_by_correctness.png'))
        plt.close()
        
        # 保存响应时间统计到CSV
        with open(os.path.join(output_dir, 'response_time_stats.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['指标', '值'])
            writer.writerow(['样本数', len(valid_df)])
            writer.writerow(['平均响应时间', f"{mean_time:.2f}"])
            writer.writerow(['中位响应时间', f"{median_time:.2f}"])
            writer.writerow(['响应时间标准差', f"{std_time:.2f}"])
            
        # 按难度分组的响应时间统计
        rt_by_difficulty = valid_df.groupby('difficulty')['response_time'].agg(['mean', 'median', 'std', 'count'])
        rt_by_difficulty.to_csv(os.path.join(output_dir, 'response_time_by_difficulty.csv'))
        
        # 按正确性分组的响应时间统计
        rt_by_correct = valid_df.groupby('is_correct')['response_time'].agg(['mean', 'median', 'std', 'count'])
        rt_by_correct.to_csv(os.path.join(output_dir, 'response_time_by_correctness.csv'))

def analyze_single_file(csv_path, output_dir):
    """
    分析单个CSV文件
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
    """
    # 读取CSV结果文件
    results_df = read_test_results(csv_path)
    if results_df is None:
        return
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理出现在数据中的方向类别
    unique_directions = set(results_df['b2a_direction'].unique()).union(set(results_df['user_answer'].unique()))
    # 排除timeout
    class_names = [d for d in unique_directions if d != 'timeout']
    
    # 分析测试结果
    analyze_by_difficulty(results_df, class_names, output_dir)
    
    # 分析响应时间
    analyze_response_times(results_df, output_dir)
    
    print(f"文件 {os.path.basename(csv_path)} 分析完成！结果已保存到目录: {output_dir}")

def main():
    """主函数"""
    # parser = argparse.ArgumentParser(description="分析空间关系测试结果")
    # parser.add_argument('--csv', type=str, required=True, help="结果CSV文件路径")
    # parser.add_argument('--output', type=str, default="./analysis_results", help="输出目录")
    
    # args = parser.parse_args()

    csv_path = "SpatialCognitionTest/Results"
    
    # 检查csv_path是文件还是文件夹
    if os.path.isdir(csv_path):
        # 获取文件夹下所有CSV文件
        csv_files = [os.path.join(csv_path, f) for f in os.listdir(csv_path) if f.endswith('.csv')]
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 创建一个空的DataFrame来保存所有数据
        all_results_df = pd.DataFrame()
        
        # 读取并合并所有CSV文件
        for csv_file in csv_files:
            df = read_test_results(csv_file)
            if df is not None:
                all_results_df = pd.concat([all_results_df, df], ignore_index=True)
        
        print(f"合并后的总数据量: {len(all_results_df)} 条")
        
        if len(all_results_df) > 0:
            # 为合并的数据创建一个输出目录
            output_dir = os.path.join(csv_path, "combined_analysis")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 处理出现在数据中的方向类别
            unique_directions = set(all_results_df['b2a_direction'].unique()).union(set(all_results_df['user_answer'].unique()))
            # 排除timeout
            class_names = [d for d in unique_directions if d != 'timeout']
            
            # 分析合并后的测试结果
            analyze_by_difficulty(all_results_df, class_names, output_dir)
            
            # 分析响应时间
            analyze_response_times(all_results_df, output_dir)
            
            print(f"\n所有数据分析完成！结果已保存到目录: {output_dir}")
        else:
            print("未成功读取任何有效数据")
    else:
        # 单个文件的情况
        output_dir = os.path.splitext(csv_path)[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        analyze_single_file(csv_path, output_dir)

if __name__ == "__main__":
    main()
