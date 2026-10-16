import pandas as pd

# 文件路径
file_path = r"..\data\enterprise_default_probabilities.csv"

# 读取数据
data = pd.read_csv(file_path)

# 提取企业代号中的数字部分，用于排序
data['enterprise_num'] = data['企业代号'].str.extract('(\d+)').astype(int)

# 按照企业代号中的数字排序
sorted_data = data.sort_values(by='enterprise_num')

# 删除临时列
sorted_data = sorted_data.drop(columns=['enterprise_num'])

# 保存排序后的数据到原文件（或新文件）
sorted_data.to_csv(file_path, index=False)

print("数据已按照企业代号排序并保存。")
