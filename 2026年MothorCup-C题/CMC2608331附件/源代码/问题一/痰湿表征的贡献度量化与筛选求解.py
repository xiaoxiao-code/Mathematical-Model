import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    data_path = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"
    df = pd.read_excel(data_path)

    # 设定目标变量和提纯出的17项关键特征
    target_col = '痰湿质'
    selected_features = [
        'ADL吃饭', 'ADL用厕', 'IADL理财', 'TC（总胆固醇）', 'IADL交通',
        'ADL洗澡', 'BMI', 'IADL购物', '空腹血糖', 'HDL-C（高密度脂蛋白）',
        'ADL步行', 'IADL服药', '血尿酸', 'LDL-C（低密度脂蛋白）',
        'TG（甘油三酯）', 'ADL穿衣', 'IADL做饭'
    ]

    # 提取特征矩阵与目标变量
    X = df[selected_features]
    y = df[target_col]

    # 构建 XGBoost 集成树模型逼近痰湿体质得分
    # 设定均方误差与正则化约束，对应结构化风险目标函数
    model = xgb.XGBRegressor(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=200,
        objective='reg:squarederror',
        random_state=42
    )

    # 通过二阶泰勒展开式局部近似优化并完成模型拟合
    model.fit(X, y)

    # 引入 TreeExplainer 进行 SHAP 博弈论归因解析
    explainer = shap.TreeExplainer(model)

    # 将模型复杂预测值线性解构为基线期望与各特征边际贡献之和
    shap_values = explainer.shap_values(X)

    # 基于 SHAP 值的痰湿特征归因蜂群图
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(shap_values, X, show=False)
    plt.title('基于 SHAP 值的痰湿体质非线性特征归因蜂群图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('SHAP_Summary_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 对特征在所有样本中的 SHAP 绝对值求平均，计算全局重要度指数
    global_importance = np.abs(shap_values).mean(axis=0)

    # 将全局重要度与特征名打包并降序排列
    importance_df = pd.DataFrame({
        '特征名称': selected_features,
        '全局重要度(I_j)': global_importance
    }).sort_values(by='全局重要度(I_j)', ascending=False).reset_index(drop=True)

    # 痰湿体质核心特征全局重要性排序柱状图
    plt.figure(figsize=(12, 8), dpi=300)
    plt.barh(importance_df['特征名称'][::-1], importance_df['全局重要度(I_j)'][::-1], color='steelblue')
    plt.xlabel('全局重要度指数', fontsize=12)
    plt.ylabel('候选特征', fontsize=12)
    plt.title('痰湿体质核心特征全局重要性排序柱状图', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('Feature_Importance_BarPlot.png', dpi=300)
    plt.close()

    # 打印结果
    for index, row in importance_df.iterrows():
        print(f"第 {index + 1} 名: {row['特征名称']} (贡献度 I_j = {row['全局重要度(I_j)']:.4f})")