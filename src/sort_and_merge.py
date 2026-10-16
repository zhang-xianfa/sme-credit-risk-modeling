import pandas as pd

# 文件路径
file1_path = r"..\data\enterprise_default_probabilities_xgboost.csv"
file2_path = r"..\data\enterprise_default_probabilities.csv"
output_path = r"..\data\merged_average.csv"

# 读取两个排序后的CSV文件
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# 确保企业代号列名一致（假设列名为'企业代号'）
# 如果列名不同，可以手动指定，例如：df1.rename(columns={'old_name': '企业代号'}, inplace=True)

# 合并两个DataFrame，基于企业代号
merged_df = pd.merge(df1, df2, on='企业代号', suffixes=('_file1', '_file2'))

# 假设预测概率列名为'预测概率'，如果不同请修改
# 计算平均值
merged_df['平均值'] = (merged_df['预测违约概率_file1'] + merged_df['预测违约概率_file2']) / 2

# 保存结果到新文件
merged_df.to_csv(output_path, index=False)

print("合并完成，结果已保存到:", output_path)
