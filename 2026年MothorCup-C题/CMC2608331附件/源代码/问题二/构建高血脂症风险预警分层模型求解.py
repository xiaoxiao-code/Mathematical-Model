import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pygam import LogisticGAM, s, f
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':

    # 路径
    FILE_PATH = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"

    if not os.path.exists(FILE_PATH):
        print(f"未在路径 {FILE_PATH} 找到数据文件。")
        exit()

    df = pd.read_excel(FILE_PATH)
    y = df['高血脂症二分类标签']

    # 构建对照组：存在标签泄露的特征矩阵
    X_leaked = df.drop(columns=['样本ID', '高血脂症二分类标签', '血脂异常分型标签（确诊病例）'], errors='ignore')

    # 构建实验组：靶向剥离后的特征矩阵
    leakage_keywords = ['HDL', 'LDL', 'TG', 'TC', '高密度', '低密度', '甘油三酯', '胆固醇']
    columns_to_strip = [col for col in X_leaked.columns if any(kw in col.upper() for kw in leakage_keywords)]
    X_stripped = X_leaked.drop(columns=columns_to_strip)


    # 根据特征唯一值分布，动态构建 B-样条基函数与分类因子
    def build_gam_terms(X_df):
        terms = None
        for i, col in enumerate(X_df.columns):
            is_categorical = X_df[col].nunique() <= 5
            current_term = f(i) if is_categorical else s(i)
            terms = current_term if terms is None else terms + current_term
        return terms


    terms_leaked = build_gam_terms(X_leaked)
    terms_stripped = build_gam_terms(X_stripped)

    # 划分数据集
    X_train_s, X_test_s, y_train, y_test = train_test_split(X_stripped, y, test_size=0.3, random_state=42)
    X_train_l, X_test_l = X_leaked.loc[X_train_s.index], X_leaked.loc[X_test_s.index]

    # 构建 Logit 联系函数与带粗糙度惩罚的似然目标函数
    gam_leaked = LogisticGAM(terms_leaked)
    gam_stripped = LogisticGAM(terms_stripped)

    # 通过 gridsearch() 执行 PIRLS 迭代，并基于 GCV 准则自适应寻优
    print("正在通过 GCV 广义交叉验证进行 PIRLS 迭代求解...")
    gam_leaked.gridsearch(X_train_l.values, y_train.values, progress=False)
    gam_stripped.gridsearch(X_train_s.values, y_train.values, progress=False)

    # 输出连续生理发病概率
    prob_leaked = gam_leaked.predict_mu(X_test_l.values)
    prob_stripped = gam_stripped.predict_mu(X_test_s.values)

    print("\n靶向剥离前后的 GAM 模型评估对比")
    print(f"[泄露组] Brier Score: {brier_score_loss(y_test, prob_leaked):.4f}")
    print(f"[剥离组] Brier Score: {brier_score_loss(y_test, prob_stripped):.4f}")

    # 靶向剥离前后的概率分布核密度对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    # 存在临床泄露的概率分布
    sns.kdeplot(prob_leaked[y_test == 0], fill=True, color='#2c7bb6', label='未确诊', ax=axes[0])
    sns.kdeplot(prob_leaked[y_test == 1], fill=True, color='#d7191c', label='确诊', ax=axes[0])
    axes[0].set_title('剥离前：临床指标泄露导致概率绝对极化', fontsize=13)
    axes[0].set_xlabel('模型预测概率', fontsize=12)
    axes[0].set_xlim(-0.1, 1.1)
    axes[0].legend()

    # 靶向剥离后的真实生理风险测度
    sns.kdeplot(prob_stripped[y_test == 0], fill=True, color='#2c7bb6', label='健康及潜在风险', ax=axes[1])
    sns.kdeplot(prob_stripped[y_test == 1], fill=True, color='#d7191c', label='高体质风险累积', ax=axes[1])
    axes[1].set_title('剥离后：GAM 拟合的真实生理风险测度', fontsize=13)
    axes[1].set_xlabel('隐性连续风险概率测度 p', fontsize=12)
    axes[1].set_xlim(-0.1, 1.1)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('KDE_Before_After_Stripping_GAM.png', dpi=300)
    plt.show()

    # 多子图并列的 GAM 偏依赖平滑曲线图
    continuous_indices = [i for i, col in enumerate(X_stripped.columns) if X_stripped[col].nunique() > 5]
    plot_indices = continuous_indices[:6] if len(continuous_indices) >= 6 else continuous_indices

    fig, axs = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axs = axs.flatten()

    for idx, feature_idx in enumerate(plot_indices):
        feature_name = X_stripped.columns[feature_idx]
        # 提取第 feature_idx 个特征的偏依赖空间网络网格
        XX = gam_stripped.generate_X_grid(term=feature_idx)
        # 计算偏依赖效应及 95% 置信区间
        pdep, confidence = gam_stripped.partial_dependence(term=feature_idx, X=XX, width=0.95)

        axs[idx].plot(XX[:, feature_idx], pdep, color='#d7191c', linewidth=2)
        axs[idx].fill_between(XX[:, feature_idx], confidence[:, 0], confidence[:, 1], color='#d7191c', alpha=0.2)
        axs[idx].set_title(f'特征: {feature_name} 的非线性偏效应', fontsize=12)
        axs[idx].set_xlabel(feature_name, fontsize=11)
        axs[idx].set_ylabel('发病风险影响', fontsize=11)
        axs[idx].grid(True, linestyle='--', alpha=0.5)

    plt.suptitle('广义加性模型揭示的中医体质与量表非线性风险驱动机制', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('GAM_Partial_Dependence_Plots.png', dpi=300)
    plt.show()

    # 结果输出
    X_out_s = X_stripped.head(1000)
    X_out_l = X_leaked.head(1000)
    y_out = y.head(1000)

    prob_leaked_1000 = gam_leaked.predict_mu(X_out_l.values)
    prob_stripped_1000 = gam_stripped.predict_mu(X_out_s.values)

    output_df = pd.DataFrame({
        '真实确诊标签': y_out,
        '泄露版无用概率': prob_leaked_1000,
        '靶向剥离真实隐性风险概率': prob_stripped_1000
    }, index=X_out_s.index)

    # 导出为 Excel 表格
    output_df.to_excel('GAM_风险概率分层.xlsx')