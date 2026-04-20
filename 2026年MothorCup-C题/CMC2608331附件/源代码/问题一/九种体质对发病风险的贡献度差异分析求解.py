import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 路径
    data_path = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"
    df = pd.read_excel(data_path)

    # 提取九种中医体质特征作为自变量矩阵 ，高血脂标签作为目标变量
    features = ['平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质']
    target = '高血脂症二分类标签'

    X = df[features]
    y = df[target]

    # 划分训练集与验证集
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 构建 XGBoost 非线性致病空间拟合基座
    model = xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=4,
        eval_metric='logloss',
        random_state=42,
        early_stopping_rounds=20
    )

    # 自适应寻找最优迭代轮数 K
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    # 引入合作博弈论构造 Shapley 价值函数
    explainer = shap.TreeExplainer(model)

    # 求解全体样本的局部边际贡献度矩阵
    shap_values = explainer.shap_values(X)

    # 整合计算每种体质的全局绝对贡献度
    global_importances = np.mean(np.abs(shap_values), axis=0)

    # 将特征名称与全局贡献度打包并降序排列
    importance_df = pd.DataFrame({
        '体质特征': features,
        '全局绝对贡献度': global_importances
    }).sort_values(by='全局绝对贡献度', ascending=True)

    # 基于 SHAP 值的九种中医体质高血脂发病风险归因蜂群图
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.title('基于 SHAP 值的九种中医体质高血脂发病风险归因蜂群图', fontsize=15, pad=15)
    plt.xlabel('SHAP 边际贡献度', fontsize=12)
    plt.tight_layout()
    plt.savefig('01_SHAP_Summary_BeeSwarm_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 九种中医体质全局绝对贡献度差异排序柱状图
    plt.figure(figsize=(10, 6), dpi=300)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(features)))
    plt.barh(importance_df['体质特征'], importance_df['全局绝对贡献度'], color=colors, edgecolor='black', alpha=0.8)

    # 在柱状图末端添加精确数值标签
    for index, value in enumerate(importance_df['全局绝对贡献度']):
        plt.text(value + 0.0005, index, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')

    plt.xlabel('全局绝对贡献度', fontsize=12)
    plt.ylabel('中医体质类型', fontsize=12)
    plt.title('九种中医体质高血脂发病风险全局绝对贡献度差异排序', fontsize=15, pad=15)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.xlim(0, importance_df['全局绝对贡献度'].max() * 1.15)
    plt.tight_layout()
    plt.savefig('02_SHAP_Global_Contribution_BarPlot.png', dpi=300)
    plt.close()

    # 逆序输出最强贡献度排名
    top_down_df = importance_df.sort_values(by='全局绝对贡献度', ascending=False)
    for rank, (idx, row) in enumerate(top_down_df.iterrows(), 1):
        print(f"第 {rank} 名: {row['体质特征']} (贡献度: {row['全局绝对贡献度']:.4f})")