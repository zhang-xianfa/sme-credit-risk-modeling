# 中小微企业信贷风险评估建模

对 2020 年高教社杯全国大学生数学建模竞赛 C 题「中小微企业的信贷决策」的完整求解：
从企业进销项发票数据出发做特征工程，用 **XGBoost / LightGBM / 决策树** 三套模型
估计企业违约概率，再融合两套强模型的输出给出最终风险排序。

## 一、问题背景

银行需要根据企业提供的进销项发票数据判断其信贷风险，决定是否放贷、放多少、利率定多少。
难点在于：

- **样本极少**：附件 1 只有 122 家有信贷记录（含标签）的企业，附件 2 的 302 家无信贷记录
  企业没有标签，属于典型的**小样本 + 半监督**场景；
- **特征要自己造**：原始数据是逐张发票的流水，需要按企业聚合成经营特征；
- **类别不平衡**：违约样本占比低，直接用准确率评估会失真。

## 二、处理流程

```
附件1（122 家有记录企业）          附件2（302 家无记录企业）
        │                                    │
        └──────────► 数据预处理.py ◄──────────┘
                  发票流水 → 企业级经营特征
                  （按年化口径统一，输出 preprocessed_data_problem1_annualized.csv）
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
       XGBoost 模型     LightGBM 模型     决策树模型
     (网格搜索 + 分层   (网格搜索 + 分层    (baseline 对照)
       交叉验证)          交叉验证)
              │              │
              └──────► sort_and_merge.py ◄──────┘
                       两模型违约概率取平均
                             │
                             ▼
                    merged_average.csv（最终风险排序）
```

**评估口径**：因为类别不平衡，评估用 **ROC-AUC / 精确率 / 召回率 / F1** 而不是准确率；
训练时用 `StratifiedKFold` 保持各折的违约比例一致，并用 `GridSearchCV` 搜超参。

**为什么要融合两个模型**：XGBoost 与 LightGBM 在同一份小样本上容易过拟合到不同的特征组合，
两者取平均能明显降低方差 —— 这也是后面 `merged_average.csv` 的由来。

## 三、目录结构

```
sme-credit-risk-modeling/
├── src/
│   ├── 爬虫.py                    # 用 baostock 拉 A 股历史行情，做企业外部经营环境的探索
│   ├── 数据预处理.py                # 发票流水 → 企业级特征（核心，374 行）
│   ├── 合并.py / 排序.py            # 多源数据合并与按风险排序
│   ├── xgboost_train_model.py     # XGBoost：网格搜索 + 分层交叉验证
│   ├── lightgbm_信贷风险模型.py      # LightGBM：同上
│   ├── 决策树训练.py                # 决策树 baseline
│   ├── sort_and_merge.py          # 两模型违约概率取平均
│   └── 数据可视化.py                # 特征分布与模型结果可视化
├── data/                          # 中间数据与模型输出（大附件未入库，见下）
├── figures/                       # 图表
├── requirements.txt
└── README.md
```

## 四、数据

**附件 1 与附件 2 没有放进仓库** —— 两个 xlsx 合计约 60 MB，放进 Git 会让 clone 变得很慢。
它们是**公开的赛题附件**，可从全国大学生数学建模竞赛官网或历年赛题资料取得，
取得后放到 `data/` 下：

```
data/
├── 附件1：122家有信贷记录企业的相关数据.xlsx     （约 20 MB）
└── 附件2：302家无信贷记录企业的相关数据.xlsx     （约 40 MB）
```

仓库里已包含的是**体积小、且能直接看出结果**的中间产物与模型输出：

| 文件 | 说明 |
|---|---|
| `附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx` | 赛题附件 3（13 KB，直接入库） |
| `preprocessed_data_problem1_annualized.csv` | 预处理后的企业级特征表 |
| `enterprise_default_probabilities_xgboost.csv` | XGBoost 预测的违约概率 |
| `enterprise_default_probabilities.csv` | LightGBM 预测的违约概率 |
| `merged_average.csv` | 两模型平均后的最终结果 |
| `history_A_stock_k_data.csv` | 爬虫拉取的 A 股历史行情 |

## 五、运行

```bash
pip install -r requirements.txt

cd src
python 数据预处理.py                # 生成企业级特征表
python xgboost_train_model.py      # XGBoost
python lightgbm_信贷风险模型.py       # LightGBM
python sort_and_merge.py           # 融合两模型输出
python 数据可视化.py                 # 出图
```

## 六、说明

- 整理时将脚本统一扁平化到 `src/`，并把其中硬编码的本机绝对路径改为指向本仓库 `data/`
  的相对路径（共 12 处），其余逻辑未改动。
- `爬虫.py` 依赖 baostock 公开接口，无需账号；该部分是企业外部经营环境的探索性分析，
  不参与主建模流程。
- 本项目是对该赛题的完整建模实现，用于方法练习与复盘。
