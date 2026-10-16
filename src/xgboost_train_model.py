import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb # 导入 XGBoost
import matplotlib.pyplot as plt
import seaborn as sns # 用于更美观的图表

# 设置支持中文的字体，确保图表能够正确显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC'] # 优先使用黑体，备选其他中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义数据文件路径
data_path = r"..\data\preprocessed_data_problem1_annualized.csv"

# --- 1. 加载预处理好的数据 ---
# 假设您的预处理数据保存在 'preprocessed_data_problem1_annualized.csv' 文件中
# 如果是其他格式（如CSV），请修改 pd.read_excel 为 pd.read_csv 等
try:
    # 请将此处路径替换为您实际的文件路径
    df = pd.read_csv(data_path)
    print("数据加载成功！")
    print(f"数据形状: {df.shape}")
    print("数据列名:", df.columns.tolist())
    print("\n数据前5行:")
    print(df.head())
except FileNotFoundError:
    print("错误：未找到指定的数据文件。请检查文件路径或文件名是否正确。")
    # 如果文件不存在，创建一个模拟的DataFrame用于演示。
    # 请注意，模拟数据仅用于代码结构演示，其结果不能代表真实情况。
    print("\n尝试创建一个模拟数据用于演示:")
    # 使用与您原始数据相似的列名和结构
    data = {
        '企业代号': [f'E{i+1}' for i in range(122)], # 模拟122家企业
        'X1_企业规模': np.random.rand(122) * 1e9,
        'X2_企业平均利润': np.random.rand(122) * 1e8 - 5e7,
        'X3_企业平均年利润率': np.random.rand(122) * 0.5,
        'X4_企业平均年利润增长率': np.random.rand(122) * 0.3 - 0.1,
        'X5_企业平均营业额增长率': np.random.rand(122) * 0.4 - 0.1,
        'X6_企业产品退货率': np.random.rand(122) * 0.1,
        'X7_信誉评级得分': [1.00]*80 + [0.75]*20 + [0.50]*10 + [0.25]*12, # 模拟评分
        'Y_是否违约': [0]*105 + [1]*17 # 模拟违约情况，制造不平衡
    }
    df = pd.DataFrame(data)
    # Add some NaNs for demonstration of imputation
    np.random.seed(42) # for reproducibility of NaNs
    for col in ['X4_企业平均年利润增长率', 'X5_企业平均营业额增长率']:
        nan_indices = np.random.choice(df.index, size=int(0.15*len(df)), replace=False)
        df.loc[nan_indices, col] = np.nan
    df['企业代号'] = df['企业代号'].astype(str)
    df['Y_是否违约'] = df['Y_是否违约'].astype(int)
    print("已创建模拟数据。请将代码中的文件名替换为您实际的文件名以使用真实数据。")

# 确保数据加载成功才能继续
if df is None:
    print("数据加载失败，程序终止。")
    exit() # 如果数据加载失败，退出程序

# --- 2. 定义特征集 (X) 和目标变量 (Y) ---
target = 'Y_是否违约'
# 排除企业代号和目标变量列
features = [col for col in df.columns if col not in ['企业代号', target]]

X = df[features]
Y = df[target]

print(f"\n用于建模的特征列 ({len(features)}列): {features}")
print(f"目标变量列: {target}")
print(f"特征集X的形状: {X.shape}")
print(f"目标变量Y的形状: {Y.shape}")
print("\n目标变量分布:")
print(Y.value_counts())
print("目标变量比例:")
print(Y.value_counts(normalize=True))


# --- 3. 进行分层数据划分 (训练集/测试集) ---
# 使用stratify参数，确保训练集和测试集中的违约/非违约比例与原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,
    test_size=0.25, # 25%作为测试集，可以根据数据量调整
    random_state=42, # 固定随机种子以保证每次运行结果一致
    stratify=Y       # 关键：进行分层抽样
)

print("\n数据已划分为训练集和测试集 (分层抽样):")
print(f"训练集X_train形状: {X_train.shape}")
print(f"测试集X_test形状: {X_test.shape}")
print("\n训练集违约分布:")
print(y_train.value_counts(normalize=True))
print("\n测试集违约分布:")
print(y_test.value_counts(normalize=True))


# --- 4. 处理缺失值 ---
# 检查划分后的数据中的缺失值
print("\n划分后数据集缺失值统计 (处理前):")
print("X_train 缺失值:\n", X_train.isnull().sum())
print("X_test 缺失值:\n", X_test.isnull().sum())

# 使用中位数填充缺失值，中位数对异常值更鲁棒
imputer = SimpleImputer(strategy='median')

# **重要：只在训练集上拟合(fit)填充器**
imputer.fit(X_train)

# **使用拟合好的填充器转换(transform)训练集和测试集**
X_train_imputed = pd.DataFrame(imputer.transform(X_train), columns=features, index=X_train.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=features, index=X_test.index)

print("\n缺失值已处理:")
print("X_train_imputed 缺失值总数:", X_train_imputed.isnull().sum().sum())
print("X_test_imputed 缺失值总数:", X_test_imputed.isnull().sum().sum())


# --- 5. 处理类别不平衡 (计算 scale_pos_weight) ---
# XGBoost可以通过scale_pos_weight参数来处理不平衡问题
# scale_pos_weight = count(负样本) / count(正样本)
# 在这里，负样本是非违约(0)，正样本是违约(1)
# 仅基于训练集计算权重
neg_count_train = (y_train == 0).sum()
pos_count_train = (y_train == 1).sum()

# 避免除以零
scale_pos_weight_value = neg_count_train / pos_count_train if pos_count_train > 0 else 1
print(f"\n根据训练集计算的 scale_pos_weight (用于XGBoost): {scale_pos_weight_value:.2f}")


# --- 6. 构建 XGBoost 模型并进行超参数调优 ---
print("\n--- 构建 XGBoost 模型并进行超参数调优 ---")

# 初始化 XGBoost 分类器
# objective='binary:logistic': 二分类问题，输出预测概率
# eval_metric='auc': 评估指标为AUC (Area Under the ROC Curve)，适合不平衡数据
# scale_pos_weight: 传入计算好的权重，处理类别不平衡
# use_label_encoder=False: 推荐设置，避免未来版本警告/错误
# random_state: 保证结果可重复性
# n_jobs=-1: 使用所有可用的CPU核心加速
xgb_model = xgb.XGBClassifier(objective='binary:logistic',
                              eval_metric='auc',
                              scale_pos_weight=scale_pos_weight_value,
                              use_label_encoder=False, # For newer versions of XGBoost
                              random_state=42, n_jobs=-1)

# 定义参数网格用于GridSearchCV，寻找最优超参数组合
# 这些参数对模型性能有重要影响，需要根据实际数据和计算资源进行调整和尝试
# n_estimators: 提升树的数量
# learning_rate: 学习率
# max_depth: 树的最大深度
# subsample: 训练样本的子采样比例
# colsample_bytree: 特征的子采样比例
# gamma: 控制树分裂的阈值
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# 使用分层K折交叉验证 (Stratified K-Fold)
# 确保交叉验证的每一折都保留了原始数据中类别比例的分布
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化 GridSearchCV
# estimator: 需要进行超参数调优的模型
# param_grid: 包含需要尝试的超参数组合的字典
# scoring='roc_auc': 模型评估指标，这里使用AUC，因为目标是预测违约概率，且数据可能不平衡
# cv: 交叉验证策略
# verbose: 控制输出的详细程度
# n_jobs: 并行运行的作业数量，-1表示使用所有可用核心
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           scoring='roc_auc',
                           cv=skf, verbose=1, n_jobs=-1)

print("\n开始进行超参数调优 (GridSearchCV, 优化目标: AUC)...")
# 在训练集上执行网格搜索，这会训练并评估多种参数组合
grid_search.fit(X_train_imputed, y_train)

print("\nGridSearchCV 找到的最优参数:")
print(grid_search.best_params_)

# best_score_ 是 GridSearchCV 在交叉验证中得到的最佳平均分数。
# 它是对模型使用最佳参数后，在训练数据分布上的泛化能力（或可视为“训练精度”）的稳健估计。
print(f"最优交叉验证 AUC 分数 (训练数据): {grid_search.best_score_:.4f}")

# 获取使用最优参数在整个训练集上重新训练好的模型
# GridSearchCV 在找到最佳参数后，会默认使用这些参数在整个训练集上训练最终模型
best_xgb_model = grid_search.best_estimator_

# 补充：计算模型在整个训练集上的准确率，作为参考（可能存在过拟合）
y_train_pred = best_xgb_model.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"模型在整个训练集上的准确率: {train_accuracy:.4f}")


# --- 7. 模型评估 (在独立的测试集上) ---
print("\n--- 在独立的测试集上评估 XGBoost 模型性能 (预测准确率及其他指标) ---")

# 在测试集上进行预测
# predict_proba[:, 1] 获取预测为正类（违约）的概率
y_pred_proba = best_xgb_model.predict_proba(X_test_imputed)[:, 1]
# predict 获取预测的类别标签 (0或1)。默认情况下，阈值通常是0.5。
# 对于不平衡数据，您可能需要根据业务需求调整这个阈值来平衡Precision和Recall。
y_pred = best_xgb_model.predict(X_test_imputed)

# 计算评估指标
# AUC Score: 衡量模型区分正负样本的能力，对于不平衡数据比准确率更重要
auc_score = roc_auc_score(y_test, y_pred_proba)
# Accuracy: 整体预测正确的比例。这就是用户询问的“预测准确率”。
# 对于类别严重不平衡的数据，高准确率可能是因为模型预测绝大多数样本为多数类。
accuracy = accuracy_score(y_test, y_pred)

print(f"测试集 AUC Score: {auc_score:.4f}")
print(f"测试集 Accuracy (预测准确率): {accuracy:.4f}")

# 计算 Precision, Recall, F1-score。这些指标对于少数类（违约=1）的评估更具参考价值。
# 检查测试集中是否存在正样本（违约）
if (y_test == 1).sum() > 0:
    precision = precision_score(y_test, y_pred) # 预测为违约的企业中，有多少是真的违约 (TP / (TP + FP))
    recall = recall_score(y_test, y_pred)     # 所有实际违约的企业中，有多少被模型正确识别 (TP / (TP + FN))
    f1 = f1_score(y_test, y_pred)       # Precision 和 Recall 的调和平均值，综合考虑查准和查全

    print(f"测试集 Precision (预测为违约的准确率): {precision:.4f}")
    print(f"测试集 Recall (识别出实际违约的比例): {recall:.4f}")
    print(f"测试集 F1-score (综合评估): {f1:.4f}")
else:
     print("\n测试集中没有实际违约样本，跳过Precision, Recall, F1计算。")


# 打印混淆矩阵
print("\n测试集混淆矩阵 (XGBoost):")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 混淆矩阵的含义 (以违约(1)为正类):
# cm[0, 0]: 实际0 (非违约), 预测0 (非违约) -> True Negative (TN) - 真负例
# cm[0, 1]: 实际0 (非违约), 预测1 (违约)   -> False Positive (FP) - 假正例 (Type I error)
# cm[1, 0]: 实际1 (违约),   预测0 (非违约) -> False Negative (FN) - 假负例 (Type II error)
# cm[1, 1]: 实际1 (违约),   预测1 (违约)   -> True Positive (TP) - 真正例

# 可视化混淆矩阵
plt.figure(figsize=(8, 6)) # 调整图表大小
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测为 非违约 (0)', '预测为 违约 (1)'],
            yticklabels=['实际为 非违约 (0)', '实际为 违约 (1)'])
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.title('测试集混淆矩阵 (XGBoost)')
plt.show()


# --- 8. 特征重要性分析 ---
print("\n特征重要性 (XGBoost):")
# importance_type='gain' 是XGBoost默认的，基于分裂带来的平均增益
feature_importances = best_xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# 按重要性降序排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, max(6, len(features) * 0.5))) # 根据特征数量调整图表高度
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('特征重要性 (XGBoost)')
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.tight_layout() # 自动调整布局，防止标签重叠
plt.show()

# ==============================================================================
# --- 以下是使用训练好的模型预测完整数据集的违约概率 ---
# ==============================================================================

print("\n--- 开始使用训练好的模型预测完整数据集的违约概率 ---")

# --- 9. 加载完整的预处理数据 (用于预测) ---
# 通常情况下，您需要对整个数据集进行预测，包括训练集中使用过的数据以及其他需要预测的数据。
# 假设 data_path 指向的文件包含所有需要预测的企业数据。
# 如果需要预测的数据在另一个文件，请修改此处路径。
try:
    # 再次加载原始完整数据，确保包含所有企业代号
    df_full = pd.read_csv(data_path)
    print("完整数据集加载成功！")
    print(f"完整数据集形状: {df_full.shape}")
except FileNotFoundError:
    print(f"错误：再次未找到文件 '{data_path}'。无法进行完整数据集预测。请检查文件路径。")
    df_full = None # 设置为 None 表示加载失败

if df_full is not None:
    # --- 10. 准备完整数据集的特征数据 ---
    # 确保特征列与训练时使用的列完全一致
    # 如果 df_full 中包含目标列（如 'Y_是否违约'），需要排除它
    X_full = df_full[features]

    # --- 11. 使用训练集上拟合好的 imputer 填充完整数据集的缺失值 ---
    # **重要：这里只能使用之前在 X_train 上 fit 好的 imputer 进行 transform**
    # 绝对不能在这里重新 fit imputer
    X_full_imputed = pd.DataFrame(imputer.transform(X_full), columns=features, index=X_full.index)

    print("\n完整数据集缺失值填充完成。")
    print(f"X_full_imputed 缺失值总数: {X_full_imputed.isnull().sum().sum()}") # 应该为0

    # --- 12. 使用训练好的模型预测违约概率 ---
    # 使用之前通过 GridSearchCV 找到并训练好的最佳模型 best_xgb_model
    # 预测违约概率 (正类，即 Y=1 的概率)
    predicted_probabilities = best_xgb_model.predict_proba(X_full_imputed)[:, 1]

    # --- 13. 将概率添加到 DataFrame 并关联企业代号 ---
    # 确保企业代号列存在于原始完整数据中
    if '企业代号' in df_full.columns:
        # 使用原始df_full的企业代号列
        results_df = df_full[['企业代号']].copy()
        results_df['预测违约概率'] = predicted_probabilities

        # --- 14. 根据违约概率对企业进行排序 ---
        results_df_sorted = results_df.sort_values(by='预测违约概率', ascending=False)

        # --- 15. 展示排序结果 ---
        print("\n企业信贷风险量化评估结果 (按预测违约概率降序排列):")
        # 打印所有企业的预测结果，方便后续使用，可以根据需要打印全部或部分
        print(results_df_sorted)

        # Optional: Save the results to a CSV file
        try:
            results_df_sorted.to_csv('enterprise_default_probabilities_xgboost.csv', index=False)
            print("\n预测违约概率结果已保存到 'enterprise_default_probabilities_xgboost.csv'")
        except Exception as e:
            print(f"\n保存文件时发生错误: {e}")

    else:
        print("错误：完整数据集中未找到 '企业代号' 列，无法关联预测结果。")

else:
    print("\n由于完整数据集加载失败，无法进行违约概率预测。请检查文件路径。")
