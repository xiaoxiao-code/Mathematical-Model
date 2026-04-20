import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def calculate_vif(X_df):
    # 构建辅助线性回归方程
    X_with_const = sm.add_constant(X_df)
    vif_data = pd.DataFrame()
    vif_data["特征"] = X_df.columns
    vif_list = []
    for i in range(1, X_with_const.shape[1]):
        # 捕捉共线性导致的除零异常
        try:
            vif_val = variance_inflation_factor(X_with_const.values, i)
        except RuntimeWarning:
            vif_val = 500
        vif_list.append(vif_val)
    vif_data["VIF"] = vif_list
    return vif_data


if __name__ == '__main__':
    data_path = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"
    df = pd.read_excel(data_path)

    # 定义目标变量痰湿质与候选特征集
    target_col = '痰湿质'
    candidate_cols = [
        'ADL用厕', 'ADL吃饭', 'ADL步行', 'ADL穿衣', 'ADL洗澡', 'ADL总分',
        'IADL购物', 'IADL做饭', 'IADL理财', 'IADL交通', 'IADL服药', 'IADL总分',
        '活动量表总分（ADL总分+IADL总分）', 'HDL-C（高密度脂蛋白）',
        'LDL-C（低密度脂蛋白）', 'TG（甘油三酯）', 'TC（总胆固醇）',
        '空腹血糖', '血尿酸', 'BMI'
    ]

    alpha = 0.05
    theta = 0.15

    # 基于秩映射的非参数相关性测度
    x_pool = []
    correlations = []
    for col in candidate_cols:
        r_s, p_val = stats.spearmanr(df[col], df[target_col])
        correlations.append({'col': col, 'r_s': r_s, 'p_val': p_val, 'abs_r_s': abs(r_s)})
        if p_val < alpha and abs(r_s) >= theta:
            x_pool.append(col)

    print(f"通过 Spearman 检验的候选池 X_pool 包含 {len(x_pool)} 个特征。")

    # 自适应降级机制
    if len(x_pool) < 5:
        print("保留全部相关性排名的指标进入候选池")
        correlations.sort(key=lambda x: x['abs_r_s'], reverse=True)
        # 提取全部候选集指标
        x_pool = [item['col'] for item in correlations]
        print(f"当前 X_pool 包含 {len(x_pool)} 个特征。")

    # 提取特征候选池矩阵供绘图与后续计算
    df_pool = df[x_pool]

    # 绘制 Spearman 秩相关系数热力图
    corr_matrix, p_matrix = stats.spearmanr(df_pool)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    annot_labels = np.empty_like(corr_matrix, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix[i, j]
            p_v = p_matrix[i, j]
            if p_v < 0.01:
                annot_labels[i, j] = f"{val:.2f}**"
            elif p_v < 0.05:
                annot_labels[i, j] = f"{val:.2f}*"
            else:
                annot_labels[i, j] = f"{val:.2f}"

    plt.figure(figsize=(12, 10), dpi=300)
    sns.heatmap(corr_matrix, mask=mask, annot=annot_labels, fmt="", cmap="coolwarm",
                vmax=1.0, vmin=-1.0, center=0, square=True, linewidths=.5,
                xticklabels=x_pool, yticklabels=x_pool, cbar_kws={"shrink": .8})
    plt.title("Spearman 秩相关系数热力图", pad=20, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('Spearman_Heatmap.png', dpi=300)
    plt.close()

    # 多源特征 VIF 诊断与自适应迭代消融
    tau = 10.0
    current_features = x_pool.copy()

    initial_vif_df = calculate_vif(df[current_features])
    initial_vif_df['VIF'] = initial_vif_df['VIF'].replace(np.inf, 5000)
    initial_vif_dict = dict(zip(initial_vif_df['特征'], initial_vif_df['VIF']))

    iteration_count = 1
    while True:
        vif_df = calculate_vif(df[current_features])
        vif_df['VIF'] = vif_df['VIF'].replace(np.inf, 5000)

        # 定位 arg max
        max_vif_idx = vif_df['VIF'].idxmax()
        max_vif_val = vif_df.loc[max_vif_idx, 'VIF']
        max_vif_feature = vif_df.loc[max_vif_idx, '特征']

        if max_vif_val > tau:
            # 从 X_pool 中降维剔除
            current_features.remove(max_vif_feature)
            print(f"迭代 {iteration_count}: 剔除冗余特征 [{max_vif_feature}], 对应 VIF = {max_vif_val:.2f}")
            iteration_count += 1
        else:
            break

    # 记录最终保留特征的 VIF
    final_vif_df = calculate_vif(df[current_features])
    final_vif_dict = dict(zip(final_vif_df['特征'], final_vif_df['VIF']))

    # 绘制多源特征 VIF 诊断与迭代消融追踪棒柱图
    features_plot = list(initial_vif_dict.keys())
    init_vifs_plot = [initial_vif_dict[f] for f in features_plot]
    final_vifs_plot = [final_vif_dict.get(f, 0) for f in features_plot]

    x_pos = np.arange(len(features_plot))
    width = 0.4

    plt.figure(figsize=(14, 7), dpi=300)
    plt.bar(x_pos - width / 2, init_vifs_plot, width, label='初始 VIF', color='salmon', alpha=0.7)
    plt.bar(x_pos + width / 2, final_vifs_plot, width, label='解耦后 VIF', color='steelblue')

    # \tau 截断阈值线
    plt.axhline(y=tau, color='red', linestyle='--', linewidth=2, label=f'容忍阈值 $\\tau$ = {tau}')

    plt.yscale('log')
    plt.ylabel('方差膨胀因子 VIF', fontsize=12)
    plt.title('多源特征 VIF 诊断与迭代消融追踪棒柱图', fontsize=16)
    plt.xticks(x_pos, features_plot, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('VIF_Iteration_Barplot.png', dpi=300)
    plt.close()

    # 打印结果
    print(f"经过 Spearman + VIF 双重防线解耦，最终提纯的关键特征共 {len(current_features)} 个：")
    for feat in current_features:
        print(f" - {feat} (最终稳定VIF: {final_vif_dict[feat]:.2f})")