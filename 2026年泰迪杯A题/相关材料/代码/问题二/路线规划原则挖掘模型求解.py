import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import warnings

# 忽略警告以保持学术输出的极简清爽
warnings.filterwarnings('ignore')

# 强制设置中文字体与负号显示，防止数据可视化图表乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 依据设定，直接读取问题一输出的特征矩阵结果
    result1_path = r"C:\Users\jack\Desktop\编写2026年泰迪杯\佐证材料\result1.xlsx"
    df_real = pd.read_excel(result1_path)

    feature_cols = ['高程', '连续坡度', '地形位置指数', '地形起伏度', '距最近河网', '距最近一级分水岭']

    # 提取真实秦直道驻点集 S_real
    X_real = df_real[feature_cols].values
    N = X_real.shape[0]

    # 基于反事实空间随机遍历假设，模拟生成数量为 N 的非秦直道控制点集
    # 通过在特征物理极值边界内进行均匀采样，构建背景对照组
    np.random.seed(42)
    X_control = np.zeros_like(X_real)
    for i in range(len(feature_cols)):
        col_min, col_max = np.min(X_real[:, i]), np.max(X_real[:, i])
        X_control[:, i] = np.random.uniform(col_min, col_max, size=N)

    # 构建全局空间特征映射矩阵
    X_global = np.vstack((X_real, X_control))
    Y_global = np.hstack((np.ones(N), np.zeros(N)))

    # 将全局张量按 8:2 比例划分为训练集与测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X_global, Y_global, test_size=0.2, random_state=42,
                                                        stratify=Y_global)

    # 构建基于 CART 架构的随机森林分类器
    rf = RandomForestClassifier(random_state=42)

    # 设定超参数网格寻优空间
    param_grid = {
        'n_estimators': [200],
        'max_depth': [12],
        'max_features': ['sqrt']
    }

    # 执行基于 K-Fold 空间分层交叉验证的网格搜索，最优化 f(X) 概率场映射函数
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    # 提取基尼指数评价准则下的最优森林集成模型
    best_rf = grid_search.best_estimator_
    print(f"随机森林模型交叉验证最高准确率: {grid_search.best_score_:.4f}")

    # 引入 TreeExplainer 极速寻优计算，突破传统 Shapley 值的时间复杂度
    explainer = shap.TreeExplainer(best_rf)

    # 计算所有特征在可能联盟序列下的边际贡献期望
    shap_values_raw = explainer.shap_values(X_test)

    # 提取正样本类的 SHAP 归因张量
    if isinstance(shap_values_raw, list):
        shap_values_positive = shap_values_raw[1]
    elif len(shap_values_raw.shape) == 3:
        shap_values_positive = shap_values_raw[:, :, 1]
    else:
        shap_values_positive = shap_values_raw

    # 计算全局特征重要度
    global_importances = np.mean(np.abs(shap_values_positive), axis=0)

    # 打印全局路线规划原则量化权重排序
    print("\n全局规划原则重要度提取")
    importance_dict = {feature_cols[i]: global_importances[i] for i in range(len(feature_cols))}
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (feat, imp) in enumerate(sorted_importances, 1):
        print(f"Rank {rank}: {feat} - 权重贡献得分: {imp:.4f}")

    # 执行数据可视化：SHAP 全局特征重要性蜂窝图
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values_positive, features=X_test, feature_names=feature_cols, show=False)
    plt.title('秦直道路线规划原则 SHAP 全局归因蜂窝图', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('SHAP_Summary_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n导出图表: SHAP_Summary_Plot.png")

    # 执行数据可视化：生成所有特征的 SHAP 单特征依赖图
    print("\n开始生成各特征的 SHAP 单特征依赖图...")
    for feature in feature_cols:
        plt.figure(figsize=(8, 6), dpi=300)
        shap.dependence_plot(feature, shap_values_positive, features=X_test, feature_names=feature_cols,
                             show=False, interaction_index=None)
        plt.title(f'{feature} - SHAP 边际贡献阈值演化图', fontsize=14, pad=20)
        plt.tight_layout()
        file_name = f'SHAP_Dependence_Plot_{feature}.png'
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"导出图表: {file_name}")