import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    data_path = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"
    df_raw = pd.read_excel(data_path)

    # 构建 17 维特征双目标评估矩阵
    metrics_data = {
        'BMI': {'SHAP': 1.9042, 'Gain': 8.21},
        'TC（总胆固醇）': {'SHAP': 1.5400, 'Gain': 1493.76},
        'LDL-C（低密度脂蛋白）': {'SHAP': 1.4181, 'Gain': 60.93},
        'TG（甘油三酯）': {'SHAP': 1.1943, 'Gain': 1730.81},
        'IADL理财': {'SHAP': 1.1838, 'Gain': 3.45},
        'HDL-C（高密度脂蛋白）': {'SHAP': 1.1712, 'Gain': 50.81},
        'ADL吃饭': {'SHAP': 1.0911, 'Gain': 6.38},
        '血尿酸': {'SHAP': 1.0706, 'Gain': 315.61},
        '空腹血糖': {'SHAP': 0.9891, 'Gain': 7.51},
        'ADL用厕': {'SHAP': 0.7696, 'Gain': 2.31},
        'IADL做饭': {'SHAP': 0.7005, 'Gain': 1.64},
        'IADL交通': {'SHAP': 0.6153, 'Gain': 0.91},
        'ADL穿衣': {'SHAP': 0.6010, 'Gain': 3.98},
        'IADL购物': {'SHAP': 0.5683, 'Gain': 2.49},
        'IADL服药': {'SHAP': 0.5263, 'Gain': 1.88},
        'ADL洗澡': {'SHAP': 0.5045, 'Gain': 1.39},
        'ADL步行': {'SHAP': 0.4530, 'Gain': 2.56}
    }

    df_metrics = pd.DataFrame.from_dict(metrics_data, orient='index')

    # 双维特征空间映射与无量纲化
    I_min, I_max = df_metrics['SHAP'].min(), df_metrics['SHAP'].max()
    W_min, W_max = df_metrics['Gain'].min(), df_metrics['Gain'].max()

    df_metrics['x'] = (df_metrics['SHAP'] - I_min) / (I_max - I_min)
    df_metrics['y'] = (df_metrics['Gain'] - W_min) / (W_max - W_min)

    X_matrix = df_metrics[['x', 'y']].values

    # 遍历 K 值计算轮廓系数
    K_range = range(2, 7)
    silhouette_scores = []

    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=42)
        labels_temp = kmeans_temp.fit_predict(X_matrix)
        score = silhouette_score(X_matrix, labels_temp)
        silhouette_scores.append(score)

    best_k = K_range[np.argmax(silhouette_scores)]

    # 基于最佳簇数自适应寻优的全局轮廓系数评估折线图
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(K_range, silhouette_scores, marker='o', color='steelblue', linewidth=2, markersize=8)
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'最佳簇数 K={best_k}')
    plt.scatter(best_k, max(silhouette_scores), color='red', s=150, zorder=5)
    plt.xlabel('聚类簇数', fontsize=12)
    plt.ylabel('全局平均轮廓系数', fontsize=12)
    plt.title('无监督特征圈层数量自适应寻优曲线', fontsize=15, pad=15)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('01_Silhouette_Score_Optimization.png', dpi=300)
    plt.close()

    # 执行最优参数下的 K-Means++ 迭代更新
    kmeans_final = KMeans(n_clusters=best_k, init='k-means++', max_iter=300, random_state=42)
    df_metrics['cluster'] = kmeans_final.fit_predict(X_matrix)
    cluster_centers = kmeans_final.cluster_centers_

    # 计算各簇质心距空间理想点的欧氏距离
    distances_to_ideal = [np.sqrt((cx - 1.0) ** 2 + (cy - 1.0) ** 2) for cx, cy in cluster_centers]
    best_cluster_idx = np.argmin(distances_to_ideal)

    # 提取隶属于理想簇的终极关键指标群
    final_key_features = df_metrics[df_metrics['cluster'] == best_cluster_idx].index.tolist()

    # 基于 K-Means++ 的双效特征空间聚类二维散点图
    plt.figure(figsize=(10, 8), dpi=300)

    # 颜色映射机制
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']

    for cluster_id in range(best_k):
        cluster_data = df_metrics[df_metrics['cluster'] == cluster_id]

        # 将最优簇单独高亮并绘制定界包络虚线圈
        if cluster_id == best_cluster_idx:
            plt.scatter(cluster_data['x'], cluster_data['y'], s=120, c='red', marker='*',
                        label='双效关键指标群')
            center_x, center_y = cluster_centers[cluster_id]
            # 绘制阴影定界圈，直观向评委展示第一象限核心萃取区
            circle = patches.Circle((center_x, center_y), radius=0.25, edgecolor='red', facecolor='red', alpha=0.1,
                                    linestyle='--')
            plt.gca().add_patch(circle)
        else:
            plt.scatter(cluster_data['x'], cluster_data['y'], s=80, c=colors[cluster_id], alpha=0.7,
                        label=f'特征次级簇 {cluster_id}')

    # 标记理想双优极值点
    plt.scatter(1.0, 1.0, color='black', marker='X', s=150, label='双优极值点')

    # 为关键特征添加文本标注
    for feature in final_key_features:
        plt.annotate(feature,
                     (df_metrics.loc[feature, 'x'], df_metrics.loc[feature, 'y']),
                     xytext=(8, -8), textcoords='offset points',
                     fontsize=10, color='darkred', fontweight='bold')

    plt.xlabel('痰湿严重程度表征度', fontsize=12)
    plt.ylabel('高血脂发病预警度', fontsize=12)
    plt.title('基于 K-Means++ 的双效特征空间聚类与萃取散点图', fontsize=16, pad=15)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(-0.1, 1.2)
    plt.ylim(-0.1, 1.2)
    plt.tight_layout()
    plt.savefig('02_KMeans_Dual_Objective_Clustering.png', dpi=300)
    plt.close()

    # 求解结果
    print(f"自适应确定最佳特征聚类群落数 K* = {best_k}")
    for idx, dist in enumerate(distances_to_ideal):
        print(f"簇 {idx} 质心距理论极值点(1,1)的空间距离: {dist:.4f}")

    print("\n锁定距极值点最近的绝对双效优等生群落，最终提纯的关键指标为：")
    for feat in final_key_features:
        print(f" -> 【{feat}】 (原始SHAP: {metrics_data[feat]['SHAP']:.4f}, 原始Gain: {metrics_data[feat]['Gain']:.2f})")