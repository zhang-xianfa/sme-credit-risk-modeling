import pandas as pd
import os
import numpy as np
from collections import defaultdict

# 导入数据处理和文件操作相关库

# 定义数据存储路径（请根据实际情况修改为您的数据文件所在目录）
# IMPORTANT: Replace with your actual data directory
data_dir = r'..\data' # <-- 请务必修改为您的实际路径

# 定义附件1文件名
attachment1_filename = '附件1：122家有信贷记录企业的相关数据.xlsx'

# 拼接完整文件路径
attachment1_path = os.path.join(data_dir, attachment1_filename)

print(f"尝试读取文件: {attachment1_path}")

# --- 数据加载 ---
try:
    # 使用pd.ExcelFile读取Excel文件（不加载全部数据到内存）
    excel_file = pd.ExcelFile(attachment1_path)
    # 获取Excel中所有工作表名称
    sheet_names = excel_file.sheet_names

    # 检查工作表数量是否符合预期（至少3个）
    if len(sheet_names) < 3:
        raise ValueError(f"附件1 Excel 文件预期包含至少3个Sheet, 但找到 {len(sheet_names)} 个。Sheet名称: {sheet_names}")

    # 按工作表名称读取数据（需确保实际表名与代码一致）
    df_company_info = excel_file.parse('企业信息')         # 企业基本信息表
    df_purchase_invoices = excel_file.parse('进项发票信息') # 进项发票表
    df_sales_invoices = excel_file.parse('销项发票信息')   # 销项发票表

    print(f"成功读取Sheet: '企业信息', '进项发票信息', '销项发票信息'")
    # print(f"实际读取到的Sheet名称列表: {sheet_names}") # 打印实际表名用于核对

except FileNotFoundError:
    # 处理文件不存在异常
    print(f"错误：文件未找到，请检查路径：{attachment1_path}")
    # Exit or handle error appropriately
    exit()
except ValueError as ve:
    # 处理工作表结构异常
    print(f"错误：{ve}")
    print("请检查附件1 Excel文件的Sheet结构和名称。")
    # Exit or handle error appropriately
    exit()
except Exception as e:
    # 处理其他未知异常
    print(f"读取Excel文件时发生错误: {e}")
    # Exit or handle error appropriately
    exit()


# --- 1. 处理企业基本信息 ---
# 重命名列（保持与后续代码一致的列名，虽然这里没有修改，但保持良好的习惯）
df_company_info = df_company_info.rename(columns={
    '企业代号': '企业代号',
    '信誉评级': '信誉评级',
    '是否违约': '是否违约'
})

# 保留所有信誉评级（包括D类）的企业数据用于建模
# 将信誉评级转换为数值（用于模型输入，A=1.0为最高，D=0.25为最低）
rating_map = {'A': 1.0, 'B': 0.75, 'C': 0.5, 'D': 0.25}
df_company_info['信誉评级得分'] = df_company_info['信誉评级'].map(rating_map)

# 将违约状态转换为数值（0=未违约，1=违约，作为模型目标变量）
df_company_info['是否违约_数值'] = df_company_info['是否违约'].apply(lambda x: 1 if x == '是' else 0)

# 获取所有企业的代号列表（包括D类）
company_ids_to_process = df_company_info['企业代号'].unique()
print(f"保留所有企业数量: {len(company_ids_to_process)}")


# --- 2. 处理发票数据 ---

# 为销项/进项发票添加类型标识（便于后续合并分析）
df_sales_invoices['发票类型'] = '销项'
df_purchase_invoices['发票类型'] = '进项'

# 统一发票数据列名（确保销项/进项表结构一致）
df_sales_invoices = df_sales_invoices.rename(columns={
    '企业代号': '企业代号',       # 企业唯一标识
    '开票日期': '开票日期',       # 发票日期
    '金额': '金额',             # 税前金额 - Not used for aggregation
    '税额': '税额',             # 增值税额 - Not used for aggregation
    '价税合计': '价税合计',       # 含税总金额 - Used for aggregation
    '发票状态': '发票状态',       # 发票状态（如正常、作废）
    '销方单位代号': '对方单位代号'  # 交易对手方代号（统一为"对方单位代号"）
})
df_purchase_invoices = df_purchase_invoices.rename(columns={
    '企业代号': '企业代号',       # 同上
    '开票日期': '开票日期',       # 同上
    '金额': '金额',             # 同上
    '税额': '税额',             # 同上
    '价税合计': '价税合计',       # 同上
    '发票状态': '发票状态',       # 同上
    '购方单位代号': '对方单位代号'  # 同上
})

# 合并销项和进项发票数据（纵向拼接）
df_invoices = pd.concat([df_sales_invoices, df_purchase_invoices], ignore_index=True)

# 剔除作废发票（仅保留有效发票）
df_invoices = df_invoices[df_invoices['发票状态'] != '作废发票'].copy()
print(f"剔除作废发票后，剩余发票数量: {len(df_invoices)}")

# 转换开票日期为日期时间类型
df_invoices['开票日期'] = pd.to_datetime(df_invoices['开票日期'])

# 提取发票年份和月份
df_invoices['年份'] = df_invoices['开票日期'].dt.year
df_invoices['月份'] = df_invoices['开票日期'].dt.month

# 仅保留2017-2019年数据
df_invoices_filtered_years = df_invoices[df_invoices['年份'].isin([2017, 2018, 2019])].copy()
print(f"保留2017-2019年发票数量: {len(df_invoices_filtered_years)}")

# 仅保留所有待处理企业（含D类）的发票数据
df_invoices_to_process = df_invoices_filtered_years[
    df_invoices_filtered_years['企业代号'].isin(company_ids_to_process)
].copy()
print(f"保留需要处理企业（含D类）发票数量: {len(df_invoices_to_process)}")


# --- 3. 月度汇总并年度化 ---

print("\n--- 开始进行月度汇总并年度化 ---")

# 步骤 3.1: 按企业-年份-月份-发票类型汇总价税合计
monthly_summary = df_invoices_to_process.groupby(['企业代号', '年份', '月份', '发票类型'])['价税合计'].sum().reset_index()

# 步骤 3.2: 按企业-年份-发票类型计算月度数量和总金额，并进行年度化
annualized_data_list = []

# Iterate through groups (company, year, invoice_type) in monthly_summary
for (company, year, invoice_type), group in monthly_summary.groupby(['企业代号', '年份', '发票类型']):
    month_count = len(group) # Number of months with data for this type in this year
    total_amount = group['价税合计'].sum() # Sum of amounts for these months

    annualized_amount = np.nan # Default to NaN if no data
    if month_count > 0:
        annualized_amount = (total_amount / month_count) * 12

    annualized_data_list.append({
        '企业代号': company,
        '年份': year,
        '发票类型': invoice_type,
        '年度化金额': annualized_amount
    })

df_annualized_summary = pd.DataFrame(annualized_data_list)


# 步骤 3.3: 处理年度负数销项金额并年度化
# Filter for negative sales invoices
df_negative_sales_2017_2019 = df_invoices_to_process[(df_invoices_to_process['发票类型'] == '销项') & (df_invoices_to_process['价税合计'] < 0)].copy()

# Group negative sales by company, year, month and sum (absolute value)
monthly_negative_sales_summary = df_negative_sales_2017_2019.groupby(['企业代号', '年份', '月份'])['价税合计'].sum().abs().reset_index(name='月度负数销项金额')

# We need the number of months with ANY sales data for the denominator for negative sales annualization
# Get month counts for 销项 (any value, positive or negative) from the original monthly_summary
# FIX: Use monthly_summary instead of df_monthly_summary
sales_month_counts = monthly_summary[monthly_summary['发票类型'] == '销项'].groupby(['企业代号', '年份'])['月份'].nunique().reset_index(name='销项月数')

# Now, annualize negative sales based on the total number of months with销项 data
annualized_negative_sales_list = []
# Iterate through groups (company, year) in monthly_negative_sales_summary
for (company, year), group in monthly_negative_sales_summary.groupby(['企业代号', '年份']):
    total_neg_amount = group['月度负数销项金额'].sum() # Sum of negative amounts for months with negative sales

    # Find the total number of months with ANY sales data for this company and year
    sales_month_count_row = sales_month_counts[(sales_month_counts['企业代号'] == company) & (sales_month_counts['年份'] == year)]
    month_count_any_sales = sales_month_count_row['销项月数'].iloc[0] if not sales_month_count_row.empty else 0

    annualized_neg_amount = np.nan # Default to NaN if no data
    if month_count_any_sales > 0: # Use month count of ANY sales for denominator
         annualized_neg_amount = (total_neg_amount / month_count_any_sales) * 12

    annualized_negative_sales_list.append({
        '企业代号': company,
        '年份': year,
        '年度化负数销项金额': annualized_neg_amount
    })

df_annualized_negative_sales = pd.DataFrame(annualized_negative_sales_list)


# 步骤 3.4: 将年度化结果合并到完整企业-年份组合上
years_to_process = [2017, 2018, 2019]
all_company_year_combinations = pd.MultiIndex.from_product([company_ids_to_process, years_to_process], names=['企业代号', '年份']).to_frame(index=False)

# 将年度化销项和进项数据整形
annualized_pivot = df_annualized_summary.pivot_table(
    index=['企业代号', '年份'],
    columns='发票类型',
    values='年度化金额'
).reset_index()

# 合并年度化汇总数据 (销项, 进项)
annual_data_complete_annualized = all_company_year_combinations.merge(
    annualized_pivot,
    on=['企业代号', '年份'],
    how='left' # Left merge to keep all combinations, resulting in NaN for missing years/types
)

# 合并年度化负数销项数据
annual_data_complete_annualized = annual_data_complete_annualized.merge(
    df_annualized_negative_sales,
    on=['企业代号', '年份'],
    how='left' # Left merge, resulting in NaN for missing years
)

# Rename columns for clarity
annual_data_complete_annualized = annual_data_complete_annualized.rename(columns={
    '销项': '年度化营业额',
    '进项': '年度化进项总额',
    '年度化负数销项金额': '年度化负数销项金额',
})

# Calculate annual profit and positive sales using annualized figures
annual_data_complete_annualized['年度化盈利'] = annual_data_complete_annualized['年度化营业额'] - annual_data_complete_annualized['年度化进项总额']
# Calculate positive sales: Total Sales (销项) - Absolute Negative Sales
# If annual_data_complete_annualized['年度化营业额'] is NaN, result is NaN.
# If annual_data_complete_annualized['年度化负数销项金额'] is NaN (meaning no negative sales data), treat as 0 for calculation
annual_data_complete_annualized['年度化正销项金额'] = annual_data_complete_annualized['年度化营业额'] - annual_data_complete_annualized['年度化负数销项金额'].fillna(0)


print("年度化汇总数据（包含NaN表示无数据年份）:")
print(annual_data_complete_annualized.head())
print(annual_data_complete_annualized.isnull().sum())


# 步骤 3.5: 转换数据格式以便计算特征 (宽格式)
# Pivot to wide format: index=企业代号, columns=(年份, Metric)
annual_summary_pivot_wide_annualized = annual_data_complete_annualized.pivot_table(
    index='企业代号',
    columns='年份',
    values=['年度化营业额', '年度化进项总额', '年度化盈利', '年度化负数销项金额', '年度化正销项金额']
)

# Flatten multi-level columns
annual_summary_pivot_wide_annualized.columns = [f'{col[0].replace("年度化", "")}_{col[1]}' for col in annual_summary_pivot_wide_annualized.columns]
annual_summary_pivot_wide_annualized = annual_summary_pivot_wide_annualized.reset_index() # Move 企业代号 back to a column


print("\n年度化汇总数据（宽格式，用于特征计算）:")
print(annual_summary_pivot_wide_annualized.head())
print(annual_summary_pivot_wide_annualized.isnull().sum())


# --- 4. 计算基于年度化汇总数据的特征指标 ---

# 定义年份列表
years = [2017, 2018, 2019]

# 创建特征存储表
df_features_annualized = annual_summary_pivot_wide_annualized[['企业代号']].copy()


# X1 (企业规模)：三年平均营业额 (销项发票金额平均值) - 基于年度化后的年度营业额
turnover_cols = [f'营业额_{year}' for year in years]
# Use mean with skipna=True to average only non-NaN years
df_features_annualized['X1_企业规模'] = annual_summary_pivot_wide_annualized[turnover_cols].mean(axis=1, skipna=True)


# X2 (企业平均利润)：三年（销项-进项发票金额）平均值 - 基于年度化后的年度盈利
profit_cols = [f'盈利_{year}' for year in years]
# Use mean with skipna=True to average only non-NaN years
df_features_annualized['X2_企业平均利润'] = annual_summary_pivot_wide_annualized[profit_cols].mean(axis=1, skipna=True)


# X3 (企业平均年利润率)：三年（利润/营业额）平均值 - 基于年度化后的年度数据
profit_margin_list = []
for year in years:
    turnover_col = f'营业额_{year}'
    profit_col = f'盈利_{year}'
    # Calculate annual profit margin. If turnover is <= 0 or NaN, result is NaN.
    annual_margin = annual_summary_pivot_wide_annualized.apply(
        lambda row: row[profit_col] / row[turnover_col] if pd.notna(row[turnover_col]) and row[turnover_col] > 0 else np.nan, axis=1
    )
    profit_margin_list.append(annual_margin)

# Combine annual margins into a temporary DataFrame
temp_margins_df_annualized = pd.concat(profit_margin_list, axis=1)
# Calculate the mean of annual margins, skipping NaNs
df_features_annualized['X3_企业平均年利润率'] = temp_margins_df_annualized.mean(axis=1, skipna=True)


# X4 (企业平均年利润增长率)：(2018年利润增长率 + 2019年利润增长率) / 2 - 基于年度化后的年度盈利
# Calculate annual profit growth rate
annual_summary_pivot_wide_annualized['利润增长率_2018'] = annual_summary_pivot_wide_annualized.apply(
    lambda row: (row['盈利_2018'] - row['盈利_2017']) / row['盈利_2017'] if pd.notna(row['盈利_2017']) and row['盈利_2017'] != 0 else np.nan, axis=1
)
annual_summary_pivot_wide_annualized['利润增长率_2019'] = annual_summary_pivot_wide_annualized.apply(
    lambda row: (row['盈利_2019'] - row['盈利_2018']) / row['盈利_2018'] if pd.notna(row['盈利_2018']) and row['盈利_2018'] != 0 else np.nan, axis=1
)
# Calculate the mean of 2018 and 2019 growth rates, skipping NaNs
df_features_annualized['X4_企业平均年利润增长率'] = annual_summary_pivot_wide_annualized[['利润增长率_2018', '利润增长率_2019']].mean(axis=1, skipna=True)


# X5 (企业平均营业额增长率)：(2018年营业额增长率 + 2019年营业额增长率) / 2 - 基于年度化后的年度营业额
# Calculate annual turnover growth rate
annual_summary_pivot_wide_annualized['营业额增长率_2018'] = annual_summary_pivot_wide_annualized.apply(
    lambda row: (row['营业额_2018'] - row['营业额_2017']) / row['营业额_2017'] if pd.notna(row['营业额_2017']) and row['营业额_2017'] != 0 else np.nan, axis=1
)
annual_summary_pivot_wide_annualized['营业额增长率_2019'] = annual_summary_pivot_wide_annualized.apply(
    lambda row: (row['营业额_2019'] - row['营业额_2018']) / row['营业额_2018'] if pd.notna(row['营业额_2018']) and row['营业额_2018'] != 0 else np.nan, axis=1
)
# Calculate the mean of 2018 and 2019 growth rates, skipping NaNs
df_features_annualized['X5_企业平均营业额增长率'] = annual_summary_pivot_wide_annualized[['营业额增长率_2018', '营业额增长率_2019']].mean(axis=1, skipna=True)


# X6 (企业产品退货率)：三年平均年度化负销项发票金额 / 三年平均年度化正销项发票金额
negative_sales_cols_annualized = [f'负数销项金额_{year}' for year in years]
positive_sales_cols_annualized = [f'正销项金额_{year}' for year in years]

# Calculate three-year average annualized negative sales, skipping NaNs
# Fill resulting NaN (meaning no negative sales data in any year) with 0
df_features_annualized['三年平均负销项金额_中间计算'] = annual_summary_pivot_wide_annualized[negative_sales_cols_annualized].mean(axis=1, skipna=True).fillna(0)

# Calculate three-year average annualized positive sales, skipping NaNs
# Fill resulting NaN (meaning no positive sales data in any year) with 0
df_features_annualized['三年平均正销项金额_中间计算'] = annual_summary_pivot_wide_annualized[positive_sales_cols_annualized].mean(axis=1, skipna=True).fillna(0)


# Calculate return rate: Avg Negative Sales / Avg Positive Sales
# If Avg Positive Sales is <= 0, result is NaN
df_features_annualized['X6_企业产品退货率'] = df_features_annualized.apply(
    lambda row: row['三年平均负销项金额_中间计算'] / row['三年平均正销项金额_中间计算'] if row['三年平均正销项金额_中间计算'] > 0 else np.nan, axis=1
)


# --- 5. 合并信誉评级和违约状态 ---
# 合并信誉评级得分（X7）和违约状态（Y）到特征表
# 使用df_company_info（包含所有企业）进行合并
df_preprocessed_problem1_annualized = df_features_annualized.merge(
    df_company_info[['企业代号', '信誉评级得分', '是否违约_数值']],
    on='企业代号',          # 按企业代号合并
    how='left'             # 左连接保留所有特征表中的企业
)

# 重命名列名为最终特征名（X7和Y）
df_preprocessed_problem1_annualized = df_preprocessed_problem1_annualized.rename(columns={
    '信誉评级得分': 'X7_信誉评级得分',
    '是否违约_数值': 'Y_是否违约'
})

# 删除中间计算列
df_preprocessed_problem1_annualized = df_preprocessed_problem1_annualized.drop(columns=['三年平均负销项金额_中间计算', '三年平均正销项金额_中间计算'])


print("\n--- 数据预处理完成（问题一，年度化处理）---")
print(f"生成特征的企业数量: {len(df_preprocessed_problem1_annualized)}")
print("前5行预处理数据:")
print(df_preprocessed_problem1_annualized.head())

# 检查缺失值（NaN表示无法计算的指标或完全无数据的年份）
print("\n各特征的缺失值数量:")
print(df_preprocessed_problem1_annualized.isnull().sum())

# 检查违约状态分布（包含D类企业）
print("\n违约状态分布 (0=未违约, 1=违约):")
print(df_preprocessed_problem1_annualized['Y_是否违约'].value_counts())

# 保存预处理数据到CSV
output_path_annualized = os.path.join(data_dir, 'preprocessed_data_problem1_annualized.csv') # 更改输出文件名以区分
df_preprocessed_problem1_annualized.to_csv(output_path_annualized, index=False)
print(f"\n预处理后的数据已保存至: {output_path_annualized}")
