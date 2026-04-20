import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# 设置全局中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def load_data(route_path, boundary_path):
    # 读取新秦直道三维坐标数据
    df_route = pd.read_excel(route_path)

    # 动态匹配列名，提取三维坐标与坡度
    col_x = [c for c in df_route.columns if 'x' in c.lower() or 'X' in c][0]
    col_y = [c for c in df_route.columns if 'y' in c.lower() or 'Y' in c][0]
    col_z = [c for c in df_route.columns if '高程' in c or 'z' in c.lower()][0]
    col_slope = [c for c in df_route.columns if '坡度' in c or 'slope' in c.lower()][0]

    coords = df_route[[col_x, col_y, col_z]].values
    slopes = df_route[col_slope].values

    # 读取甘陕八县边界数据用于可视化
    xls_bounds = pd.ExcelFile(boundary_path)
    boundaries = {}
    for sheet in xls_bounds.sheet_names:
        boundaries[sheet] = pd.read_excel(boundary_path, sheet_name=sheet).values

    return coords, slopes, boundaries


def calculate_manifold_features(coords, slopes, alpha, beta, kappa):
    n_points = len(coords)
    s = np.zeros(n_points)
    delta_z_ds = np.zeros(n_points)

    # 对应公式: 计算三维空间轨迹的一维流形参数化累积弧长
    for i in range(1, n_points):
        dx = coords[i, 0] - coords[i - 1, 0]
        dy = coords[i, 1] - coords[i - 1, 1]
        dz = coords[i, 2] - coords[i - 1, 2]
        ds = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        s[i] = s[i - 1] + ds

        # 对应公式中部分项: 计算相对高程突变量导数
        if ds > 0:
            delta_z_ds[i] = dz / ds
        else:
            delta_z_ds[i] = 0

    # 对应公式: 构建局部地势咽喉特征信号
    i_throat = alpha * np.exp(kappa * np.abs(slopes)) + beta * delta_z_ds

    return s, i_throat


def extract_and_cluster_passes(s, i_throat, peak_distance, eps_dbscan, min_samples):
    # 计算地形突起度动态阈值
    prominence_threshold = 0.75 * np.std(i_throat)

    # 对应公式: 基于形态学滤波的信号极值寻峰，获取候选关隘集合
    # 转换距离单位为近似的数组索引步长
    avg_step = np.mean(np.diff(s))
    width_indices = max(1, int(peak_distance / avg_step))

    peak_indices, _ = find_peaks(i_throat, distance=width_indices, prominence=prominence_threshold)

    if len(peak_indices) == 0:
        return np.array([])

    s_peaks = s[peak_indices].reshape(-1, 1)

    # 对应 DBSCAN 理论: 基于一维里程绝对距离执行战术空间聚类
    clustering = DBSCAN(eps=eps_dbscan, min_samples=min_samples).fit(s_peaks)
    labels = clustering.labels_

    final_pass_indices = []

    # 对应公式: 对独立簇 C_k 提取咽喉指数极大值作为防御核心
    for k in set(labels):
        if k == -1:
            continue
        cluster_mask = (labels == k)
        cluster_peak_indices = peak_indices[cluster_mask]

        best_idx = cluster_peak_indices[np.argmax(i_throat[cluster_peak_indices])]
        final_pass_indices.append(best_idx)

    return np.sort(final_pass_indices)


def plot_1d_signal_and_clustering(s, i_throat, peak_indices, final_indices, eps_dbscan):
    # 绘制：一维流形特征信号演化与咽喉聚类示意图
    plt.figure(figsize=(12, 5))
    plt.plot(s, i_throat, c='steelblue', lw=1, alpha=0.8, label='咽喉特征信号')

    # 绘制初始候选波峰
    plt.scatter(s[peak_indices], i_throat[peak_indices], c='orange', s=20, label='初始候选极值驻点')

    # 渲染 DBSCAN 战区色带及最终关隘
    for idx in final_indices:
        s_val = s[idx]
        plt.axvspan(s_val - eps_dbscan / 2, s_val + eps_dbscan / 2, color='red', alpha=0.1)
        plt.scatter(s_val, i_throat[idx], c='red', marker='*', s=150, zorder=5,
                    label='融合重构最终关隘' if idx == final_indices[0] else "")

    plt.title('一维流形特征信号演化与 DBSCAN 咽喉聚类示意图')
    plt.xlabel('空间演化累积里程')
    plt.ylabel('地势咽喉指数')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_3d_macro_terrain(coords, final_indices):
    # 绘制：三维宏观地形与最终关隘卡位映射图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    ax.plot(x, y, z, c='gray', lw=1.5, alpha=0.6, label='新秦直道三维空间轨迹')

    pass_coords = coords[final_indices]
    ax.scatter(pass_coords[:, 0], pass_coords[:, 1], pass_coords[:, 2],
               c='red', marker='v', s=100, depthshade=False, label='重规划关隘')

    ax.set_title('三维宏观地形轨迹与最终关隘卡位映射图')
    ax.set_xlabel('X 坐标')
    ax.set_ylabel('Y 坐标')
    ax.set_zlabel('绝对高程')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_2d_county_map(coords, final_indices, boundaries):
    # 绘制：甘陕八县新秦直道路线与重规划关隘分布图
    plt.figure(figsize=(10, 10))

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(boundaries)))
    for idx, (name, bounds) in enumerate(boundaries.items()):
        plt.fill(bounds[:, 0], bounds[:, 1], color=colors[idx], alpha=0.6, edgecolor='gray',
                 label=name if idx < 8 else "")
        cx, cy = np.mean(bounds[:, 0]), np.mean(bounds[:, 1])
        plt.text(cx, cy, name, ha='center', va='center', fontsize=9)

    plt.plot(coords[:, 0], coords[:, 1], c='darkred', lw=2.5, label='新秦直道干线')

    pass_coords = coords[final_indices]
    plt.scatter(pass_coords[:, 0], pass_coords[:, 1], c='red', marker='^', edgecolors='black', s=120,
                label='重规划关隘战区', zorder=10)

    plt.title('甘陕八县新秦直道线路与重构关隘空间分布图')
    plt.xlabel('经向投影坐标 X / m')
    plt.ylabel('纬向投影坐标 Y / m')
    plt.legend(loc='lower right')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 定义绝对数据源路径
    ROUTE_PATH = r"C:\Users\jack\Desktop\编写2026年泰迪杯\佐证材料\新秦直道的路线坐标.xlsx"
    BOUNDS_PATH = r"C:\Users\jack\Desktop\2026年泰迪杯赛题与优秀论文\A题\全部数据\正式数据\附件3  甘陕八县的县界数据.xlsx"

    # 超参数声明
    ALPHA = 0.5
    BETA = 0.5
    KAPPA = 0.1
    PEAK_DISTANCE = 1000
    EPS_DBSCAN = 5000
    MIN_SAMPLES = 1

    print("正在加载路线与县界基底数据")
    coords, slopes, boundaries = load_data(ROUTE_PATH, BOUNDS_PATH)

    print("正在进行空间流形降维与咽喉指数计算")
    s, i_throat = calculate_manifold_features(coords, slopes, ALPHA, BETA, KAPPA)

    print("正在执行信号形态学寻峰与 DBSCAN 战区融合")
    avg_step = np.mean(np.diff(s))
    width_indices = max(1, int(PEAK_DISTANCE / avg_step))
    peak_indices, _ = find_peaks(i_throat, distance=width_indices, prominence=0.75 * np.std(i_throat))
    final_pass_indices = extract_and_cluster_passes(s, i_throat, PEAK_DISTANCE, EPS_DBSCAN, MIN_SAMPLES)

    print("\n核心求解结果")
    print(f"初始提取局部极值驻点数量：{len(peak_indices)} 个")
    print(f"经 DBSCAN 聚类去重后，全局最优关隘布设极值：{len(final_pass_indices)} 座")
    print(f"提取的最优关隘空间节点序号序列：{list(final_pass_indices)}")

    print("生成一维流形特征信号图")
    plot_1d_signal_and_clustering(s, i_throat, peak_indices, final_pass_indices, EPS_DBSCAN)

    print("生成三维宏观地形卡位映射图")
    plot_3d_macro_terrain(coords, final_pass_indices)

    print("生成甘陕八县二维底图耦合可视化")
    plot_2d_county_map(coords, final_pass_indices, boundaries)
    print("全流程执行完毕。")