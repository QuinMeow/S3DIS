'''
移除CSV文件中b2a_direction类别为Unknown的记录
'''
import pandas as pd
import argparse
import os

def remove_unknown_direction(input_file, output_file=None):
    """
    移除CSV文件中b2a_direction为Unknown的行
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径，默认为在原文件名基础上加上"_filtered"
    
    返回:
        处理后的CSV文件路径
    """
    # 检查文件是否存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到文件: {input_file}")
    
    # 默认输出文件名
    if output_file is None:
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_filtered{file_ext}"
    
    # 读取CSV文件
    print(f"正在读取CSV文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 输出原始数据信息
    total_rows = len(df)
    print(f"原始数据总行数: {total_rows}")
    
    if 'b2a_direction' not in df.columns:
        print("警告: CSV文件中没有'b2a_direction'列，无法进行过滤")
        return input_file
    
    # 统计各类别的数量
    direction_counts = df['b2a_direction'].value_counts()
    print("原始方向类别统计:")
    print(direction_counts)
    
    unknown_count = direction_counts.get('Unknown', 0)
    if unknown_count == 0:
        print("CSV文件中没有b2a_direction为'Unknown'的行")
        return input_file
    
    # 过滤掉b2a_direction为Unknown的行
    filtered_df = df[df['b2a_direction'] != 'Unknown']
    
    # 输出过滤后的数据信息
    filtered_rows = len(filtered_df)
    removed_rows = total_rows - filtered_rows
    print(f"已移除 {removed_rows} 行 (b2a_direction = 'Unknown')")
    print(f"过滤后数据行数: {filtered_rows}")
    
    # 保存过滤后的数据
    filtered_df.to_csv(output_file, index=False)
    print(f"已将过滤后的数据保存至: {output_file}")
    
    # 过滤后类别统计
    new_direction_counts = filtered_df['b2a_direction'].value_counts()
    print("过滤后方向类别统计:")
    print(new_direction_counts)
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='移除CSV文件中b2a_direction为Unknown的行')
    parser.add_argument('input_file', type=str, help='输入CSV文件路径')
    parser.add_argument('-o', '--output', type=str, default=None, help='输出CSV文件路径')
    args = parser.parse_args()
    
    try:
        output_file = remove_unknown_direction(args.input_file, args.output)
        print(f"处理完成: {output_file}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()