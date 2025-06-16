'''
移除CSV文件中b2a_direction类别为Unknown与角度差小于5度的记录
'''
import pandas as pd
import argparse
import os

def remove_unknown_direction(input_file, output_file=None, diff_bias=5):
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

    # 进一步移除yaw_diff和pitch_diff小于5的样本
    if 'yaw_diff' in filtered_df.columns and 'pitch_diff' in filtered_df.columns:
        before_angle_filter = len(filtered_df)
        filtered_df = filtered_df[(filtered_df['yaw_diff'].abs() > diff_bias) | (filtered_df['pitch_diff'].abs() > diff_bias)]
        after_angle_filter = len(filtered_df)
        print(f"已移除 {before_angle_filter - after_angle_filter} 行 (yaw_diff或pitch_diff < {diff_bias})")
    else:
        print("警告: CSV文件中缺少'yaw_diff'或'pitch_diff'列，未进行角度过滤")

    # 输出过滤后的数据信息
    filtered_rows = len(filtered_df)
    removed_rows = total_rows - filtered_rows
    print(f"已移除 {removed_rows} 行 (b2a_direction = 'Unknown' 或 yaw_diff/pitch_diff < {diff_bias})")
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
    
    input_path = "ImagePairsFromPano/5classes_dataset/test.csv"
    diff_bias = 3
    output_path = f"ImagePairsFromPano/5classes_dataset/test_No_Unkown_and_<{diff_bias}.csv"

    try:
        output_file = remove_unknown_direction(input_path, output_path, diff_bias)
        print(f"处理完成: {output_file}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()