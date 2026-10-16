import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

# 读取数据
df = pd.read_excel("问题一数据处理.xlsx")
X = df[["评级得分", "公司年平均营业额", "年平均盈利", "销售额平均增长率", "三年利润增长率", "产品退货率", "企业规模"]]
Y = df["违约"]

# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

# 构建决策树模型（最大深度8，叶子节点最小样本数20）
clf = tree.DecisionTreeClassifier(max_depth=8, min_samples_leaf=20)
clf.fit(X_train, y_train)

# 预测与评估
test_rec = X_test.iloc[1, :]  # 测试单条样本
prediction = clf.predict([test_rec])
accuracy = accuracy_score(y_test, clf.predict(X_test))
print(f"预测结果: {prediction}, 准确率: {accuracy:.4f}")