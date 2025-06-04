import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/train.csv')

# 随机抽取100条
sample_df = df.sample(n=100, random_state=42)

# 保存到新文件
sample_df.to_csv('/data2/zyz/S3DIS/ImagePairsFromPano/5classes_dataset/train_100.csv', index=False)