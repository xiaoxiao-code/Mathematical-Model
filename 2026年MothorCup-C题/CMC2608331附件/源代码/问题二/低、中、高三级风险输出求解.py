import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':

    # 路径
    raw_data_path = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"
    gam_prob_path = r"C:\Users\jack\Desktop\GAM_风险概率分层.xlsx"

    if not (os.path.exists(raw_data_path) and os.path.exists(gam_prob_path)):
        print("未找到指定路径下的数据文件，请检查绝对路径是否正确。")
        exit()

    df_raw = pd.read_excel(raw_data_path)
    df_prob = pd.read_excel(gam_prob_path)

    # 通过样本ID将连续风险概率与原始多维特征进行横向合并
    df_merged = pd.merge(df_prob, df_raw[['样本ID', '痰湿质', '活动量表总分（ADL总分+IADL总分）']], on='样本ID',
                         how='inner')

    # 提取患者的底层隐性风险概率测度
    p_array = df_merged['靶向剥离真实隐性风险概率'].values

    # 采用高斯核函数重构概率密度
    kde = gaussian_kde(p_array, bw_method='silverman')

    # 在 [0, 1] 区间内均匀生成 N=10000 个高精度连续评估格点
    p_grid = np.linspace(0, 1, 10000)

    # 获取连续拓扑密度泛函
    pdf_values = kde(p_grid)

    # 对密度曲线求一阶导数等于0且二阶导数大于0，提取极小值
    valleys, properties = find_peaks(-pdf_values, prominence=0.01)

    # 提取密度泛函的波峰
    peaks, _ = find_peaks(pdf_values, prominence=0.01)

    # 兜底截断策略
    if len(valleys) < 2:
        valleys, properties = find_peaks(-pdf_values, prominence=0.001)

    if len(valleys) >= 2:
        top_valleys_idx = sorted(valleys[np.argsort(properties['prominences'])[-2:]])
        th1 = p_grid[top_valleys_idx[0]]
        th2 = p_grid[top_valleys_idx[1]]
    else:
        th1 = np.percentile(p_array, 33.33)
        th2 = np.percentile(p_array, 66.67)

    # 构建绝对风险分层判定方程，执行自然分层
    conditions = [
        (df_merged['靶向剥离真实隐性风险概率'] <= th1),
        (df_merged['靶向剥离真实隐性风险概率'] > th1) & (df_merged['靶向剥离真实隐性风险概率'] <= th2),
        (df_merged['靶向剥离真实隐性风险概率'] > th2)
    ]
    labels = ['低风险', '中风险', '高风险']
    numeric_labels = [0, 1, 2]

    df_merged['风险等级'] = np.select(conditions, labels, default='Unknown')
    df_merged['风险类别编号'] = np.select(conditions, numeric_labels, default=-1)

    # 打印结果
    print(f"微积分极值寻优收敛完成")
    print(f"自适应判定的低-中风险波谷阈值: {th1:.4f}")
    print(f"自适应判定的中-高风险波谷阈值: {th2:.4f}")
    print("\n三级风险人群统计分布：")
    print(df_merged['风险等级'].value_counts())

    # 基于微积分寻峰的连续风险概率核密度波谷截断全景图
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # 绘制非参数概率密度空间重构曲线
    ax1.plot(p_grid, pdf_values, color='black', linewidth=2, label='风险概率核密度泛函')

    # 标出高耸的波峰
    ax1.plot(p_grid[peaks], pdf_values[peaks], "r*", markersize=10, label='拓扑集聚波峰')

    # 绘制天然波谷边界
    ax1.axvline(x=th1, color='red', linestyle='--', linewidth=2, label=f'波谷截断阈值 1 ($p={th1:.2f}$)')
    ax1.axvline(x=th2, color='darkred', linestyle='--', linewidth=2, label=f'波谷截断阈值 2 ($p={th2:.2f}$)')

    # 分块填充低、中、高风险过渡区域颜色
    ax1.fill_between(p_grid, pdf_values, where=(p_grid <= th1), color='#2c7bb6', alpha=0.3, label='低风险区')
    ax1.fill_between(p_grid, pdf_values, where=(p_grid > th1) & (p_grid <= th2), color='#ffffbf', alpha=0.5,
                     label='中风险区')
    ax1.fill_between(p_grid, pdf_values, where=(p_grid > th2), color='#d7191c', alpha=0.3, label='高风险区')

    ax1.set_xlabel('连续风险概率测度', fontsize=12)
    ax1.set_ylabel('概率密度估计值', fontsize=12)
    ax1.set_title('基于 KDE 拓扑微积分极值寻优的三级风险自然分层模型', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('KDE_Valley_Truncation_Panorama.png', dpi=300)
    plt.show()

    # 三级风险标签降维后的三维空间散点分布图
    fig2 = plt.subplots(figsize=(10, 8), dpi=300)
    ax2 = plt.axes(projection='3d')

    x_val = df_merged['痰湿质'].values
    y_val = df_merged['活动量表总分（ADL总分+IADL总分）'].values
    z_val = df_merged['靶向剥离真实隐性风险概率'].values
    c_val = df_merged['风险类别编号'].values

    # 自定义低、中、高风险颜色映射
    colors = ['#2c7bb6', '#fdae61', '#d7191c']
    for i, target_class in enumerate([0, 1, 2]):
        indices = c_val == target_class
        ax2.scatter(x_val[indices], y_val[indices], z_val[indices],
                    c=colors[i], label=labels[i], s=30, alpha=0.7, edgecolor='k', linewidth=0.2)

    ax2.set_xlabel('中医痰湿体质积分', fontsize=11)
    ax2.set_ylabel('日常生活能力量表总分', fontsize=11)
    ax2.set_zlabel('底层生理发病风险概率', fontsize=11)
    ax2.set_title('三级风险聚类人群在医疗特征空间的三维拓扑分布', fontsize=14)
    ax2.legend(title="风险自然分层标签")

    # 调整视角以最佳呈现三维分层效果
    ax2.view_init(elev=20, azim=135)
    plt.tight_layout()
    plt.savefig('3D_Risk_Scatter_Distribution.png', dpi=300)
    plt.show()

    # 导出表格
    df_merged.to_excel('三级风险分层结果.xlsx', index=False)