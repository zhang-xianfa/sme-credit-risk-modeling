import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# 设置支持中文的字体
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# 定义数据文件路径
data_path = r"..\data\preprocessed_data_problem1_annualized.csv"

# --- 1. 加载预处理好的数据 (用于训练和测试) ---
# 尝试加载用户提供的文件路径的数据
try:
    df_train_eval = pd.read_csv(data_path)
    print("用于训练和评估的数据加载成功！")
    print(f"数据形状: {df_train_eval.shape}")
    print("数据列名:", df_train_eval.columns.tolist())
    print("\n数据前5行:")
    print(df_train_eval.head())
except FileNotFoundError:
    print(f"错误：未找到文件 '{data_path}'。请检查文件路径或文件名是否正确。")
    # 如果文件不存在，创建一个模拟数据用于演示模型训练部分
    print("\n尝试创建一个模拟数据用于演示模型训练:")
    data = {
        '企业代号': [f'E{i+1}' for i in range(100)],
        'X1_企业规模': np.random.rand(100) * 1e9,
        'X2_企业平均利润': np.random.rand(100) * 1e8 - 5e7,
        'X3_企业平均年利润率': np.random.rand(100) * 0.5,
        'X4_企业平均年利润增长率': np.random.rand(100) * 0.3 - 0.1,
        'X5_企业平均营业额增长率': np.random.rand(100) * 0.4 - 0.1,
        'X6_企业产品退货率': np.random.rand(100) * 0.1,
        'X7_信誉评级得分': [1.00]*85 + [0.75]*10 + [0.50]*5, # 模拟评分
        'Y_是否违约': [0]*95 + [1]*5 # 模拟不平衡目标变量
    }
    df_train_eval = pd.DataFrame(data)
    # Add some NaNs for demonstration of imputation
    np.random.seed(42) # for reproducibility of NaNs
    for col in ['X4_企业平均年利润增长率', 'X5_企业平均营业额增长率']:
        nan_indices = np.random.choice(df_train_eval.index, size=int(0.15*len(df_train_eval)), replace=False)
        df_train_eval.loc[nan_indices, col] = np.nan
    df_train_eval['企业代号'] = df_train_eval['企业代号'].astype(str)
    df_train_eval['Y_是否违约'] = df_train_eval['Y_是否违约'].astype(int)
    print("已创建模拟数据用于模型训练。请将代码中的文件名替换为您实际的文件名以使用真实数据。")

# --- 2. 定义特征集 (X) 和目标变量 (Y) ---
target = 'Y_是否违约'
# 排除企业代号和目标变量列
features = [col for col in df_train_eval.columns if col not in ['企业代号', target]]

X = df_train_eval[features]
Y = df_train_eval[target]

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
# LightGBM/XGBoost可以通过scale_pos_weight参数来处理不平衡问题
# scale_pos_weight = count(负样本) / count(正样本)
# 在这里，负样本是非违约(0)，正样本是违约(1)
scale_pos_weight_value = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n根据训练集计算的 scale_pos_weight: {scale_pos_weight_value:.2f}")


# --- 6. 构建 LightGBM 模型并进行超参数调优 ---
# 初始化 LightGBM 分类器
# objective='binary': 二分类问题
# metric='auc': 评估指标为AUC，适合不平衡数据
# is_unbalance=False: 当使用 scale_pos_weight 时，通常将此设为 False
# scale_pos_weight: 传入计算好的权重
# random_state: 保证结果可重复
# n_jobs=-1: 使用所有可用的CPU核心加速
lgbm = lgb.LGBMClassifier(objective='binary', metric='auc',
                          is_unbalance=False,
                          scale_pos_weight=scale_pos_weight_value,
                          random_state=42, n_jobs=-1)

# 定义参数网格用于GridSearchCV，寻找最优超参数组合
# 这些参数会显著影响模型性能，需要根据实际情况调整范围
# num_leaves: 控制树的复杂度，值越大模型越复杂，可能过拟合
# max_depth: 树的最大深度，-1表示无限制
# learning_rate: 学习率，控制每一步迭代的步长
# n_estimators: 弱学习器（树）的数量
# subsample: 每次迭代时用于训练的样本比例，防止过拟合
# colsample_bytree: 每次迭代时用于训练的特征比例，防止过拟合
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [15, 31],
    'max_depth': [-1, 5, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    # 'reg_alpha': [0, 0.1, 0.5], # L1正则化
    # 'reg_lambda': [0, 0.1, 0.5] # L2正则化
}

# 使用分层K折交叉验证 (Stratified K-Fold)
# n_splits: 交叉验证的折数
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 初始化 GridSearchCV
# estimator: 要调优的模型
# param_grid: 参数网格
# scoring='roc_auc': 评估指标，选择AUC，适合不平衡数据
# cv: 交叉验证策略
# verbose: 控制输出信息量
# n_jobs: 并行使用的核心数
grid_search = GridSearchCV(estimator=lgbm, param_grid=param_grid,
                           scoring='roc_auc',
                           cv=skf, verbose=1, n_jobs=-1)

print("\n开始进行超参数调优 (GridSearchCV, 优化目标: AUC)...")
# 在训练集上执行网格搜索
grid_search.fit(X_train_imputed, y_train)

print("\nGridSearchCV 找到的最优参数:")
print(grid_search.best_params_)
# best_score_ 是交叉验证在训练集上的平均分数，可以看作是模型在训练数据分布上的“训练精度”的稳健估计
print(f"最优交叉验证 AUC 分数 (训练集): {grid_search.best_score_:.4f}")

# 获取使用最优参数训练好的模型
best_lgbm_model = grid_search.best_estimator_

# 补充：计算模型在整个训练集上的准确率
y_train_pred = best_lgbm_model.predict(X_train_imputed)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"模型在整个训练集上的准确率: {train_accuracy:.4f}")


# --- 7. 模型评估 (在独立的测试集上) ---
print("\n--- 在独立的测试集上评估模型性能 (预测准确率及其他指标) ---")

# 在测试集上进行预测
# predict_proba[:, 1] 获取预测为正类（违约）的概率
y_pred_proba = best_lgbm_model.predict_proba(X_test_imputed)[:, 1]
# predict 获取预测的类别标签 (0或1)。默认阈值通常是0.5
y_pred = best_lgbm_model.predict(X_test_imputed)

# 计算评估指标
# AUC Score: 衡量模型区分正负样本的能力，对于不平衡数据非常重要
auc_score = roc_auc_score(y_test, y_pred_proba)
# Accuracy: 整体预测正确的比例。对于不平衡数据可能不是最佳指标
accuracy = accuracy_score(y_test, y_pred)
# Precision (查准率): 预测为违约的企业中，有多少是真的违约 (TP / (TP + FP))
# Recall (查全率 / 敏感度): 所有实际违约的企业中，有多少被模型正确识别 (TP / (TP + FN))
# F1-score: Precision 和 Recall 的调和平均值，综合考虑查准和查全
# 为了计算精确率、召回率和F1，需要一个阈值将概率转换为类别。
# 这里使用 predict() 的结果，它基于默认阈值（通常为0.5）。
# 对于不平衡数据，默认的0.5阈值可能不是最优的。在实际应用中，可能需要根据业务目标调整阈值。
# 添加检查以避免测试集中没有正样本（违约）时计算指标出错
if (y_test == 1).sum() > 0:
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"测试集 AUC Score: {auc_score:.4f}")
    print(f"测试集 Accuracy: {accuracy:.4f}") # 这就是用户询问的“预测准确率”
    print(f"测试集 Precision (预测为违约的准确率): {precision:.4f}")
    print(f"测试集 Recall (识别出实际违约的比例): {recall:.4f}")
    print(f"测试集 F1-score (综合评估): {f1:.4f}")
else:
     print(f"测试集 AUC Score: {auc_score:.4f}")
     print(f"测试集 Accuracy: {accuracy:.4f}") # 这就是用户询问的“预测准确率”
     print("\n测试集中没有实际违约样本，跳过Precision, Recall, F1计算。")


# 打印混淆矩阵
print("\n测试集混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 混淆矩阵的含义:
# cm[0, 0]: 实际0 (非违约), 预测0 (非违约) -> True Negative (TN)
# cm[0, 1]: 实际0 (非违约), 预测1 (违约)   -> False Positive (FP) - Type I error
# cm[1, 0]: 实际1 (违约),   预测0 (非违约) -> False Negative (FN) - Type II error
# cm[1, 1]: 实际1 (违约),   预测1 (违约)   -> True Positive (TP)

# 可视化混淆矩阵
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测为 非违约 (0)', '预测为 违约 (1)'],
            yticklabels=['实际为 非违约 (0)', '实际为 违约 (1)'])
plt.xlabel('预测类别')
plt.ylabel('实际类别')
plt.title('测试集混淆矩阵')
plt.show()


# --- 8. 特征重要性分析 ---
print("\n特征重要性:")
feature_importances = best_lgbm_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# 按重要性降序排序
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('特征重要性')
plt.xlabel('重要性得分')
plt.ylabel('特征')
plt.tight_layout() # 自动调整布局，防止标签重叠
plt.show()

# ==============================================================================
# --- 以下是使用训练好的模型预测完整数据集的违约概率 ---
# ==============================================================================

print("\n--- 开始预测完整数据集的违约概率 ---")

# --- 9. 加载完整的预处理数据 (用于预测) ---
# 注意：这里需要加载包含所有企业的数据，包括训练集中使用过的数据，
# 以及附件1中没有Y标签的企业（如果有的话）。
# 假设 data_path 指向的是包含所有122家企业的数据文件。
try:
    df_full = pd.read_csv(data_path)
    print("完整数据集加载成功！")
    print(f"完整数据集形状: {df_full.shape}")
except FileNotFoundError:
    print(f"错误：再次未找到文件 '{data_path}'。无法进行完整数据集预测。请检查文件路径。")
    df_full = None # 设置为 None 表示加载失败

if df_full is not None:
    # --- 10. 准备完整数据集的特征数据 ---
    # 确保特征列与训练时一致
    X_full = df_full[features]

    # --- 11. 使用训练集上拟合好的 imputer 填充完整数据集的缺失值 ---
    # 使用之前在 X_train 上 fit 好的 imputer
    X_full_imputed = pd.DataFrame(imputer.transform(X_full), columns=features, index=X_full.index)

    print("\n完整数据集缺失值填充完成。")
    print("X_full_imputed 缺失值总数:", X_full_imputed.isnull().sum().sum())

    # --- 12. 使用训练好的模型预测违约概率 ---
    # 使用之前通过 GridSearchCV 找到的最佳模型 best_lgbm_model
    # 预测违约概率 (正类，即 Y=1 的概率)
    predicted_probabilities = best_lgbm_model.predict_proba(X_full_imputed)[:, 1]

    # --- 13. 将概率添加到 DataFrame 并关联企业代号 ---
    # 确保企业代号列存在
    if '企业代号' in df_full.columns:
        results_df = df_full[['企业代号']].copy()
        results_df['预测违约概率'] = predicted_probabilities

        # --- 14. 根据违约概率对企业进行排序 ---
        results_df_sorted = results_df.sort_values(by='预测违约概率', ascending=False)

        # --- 15. 展示排序结果 ---
        print("\n企业信贷风险量化评估结果 (按预测违约概率降序排列):")
        # 打印所有企业的预测结果，方便后续使用，可以只打印部分或全部
        print(results_df_sorted)

        # Optional: Save the results to a CSV file
        try:
            results_df_sorted.to_csv('enterprise_default_probabilities.csv', index=False)
            print("\n预测违约概率结果已保存到 'enterprise_default_probabilities.csv'")
        except Exception as e:
            print(f"\n保存文件时发生错误: {e}")

    else:
        print("错误：完整数据集中未找到 '企业代号' 列，无法关联预测结果。")

else:
    print("\n由于完整数据集加载失败，无法进行违约概率预测。请检查文件路径。")

