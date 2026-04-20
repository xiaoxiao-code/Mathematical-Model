import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import math
import os

# 设置全局中文字体与高分辨率，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300


def load_and_preprocess_data(route_path, boundary_path, sample_step):
    # 读取新秦直道三维坐标数据
    df_route = pd.read_excel(route_path)

    # 提取 x, y, z 坐标列，假设常用命名规则
    col_x = [c for c in df_route.columns if 'x' in c.lower() or 'X' in c][0]
    col_y = [c for c in df_route.columns if 'y' in c.lower() or 'Y' in c][0]
    col_z = [c for c in df_route.columns if '高程' in c or 'z' in c.lower()][0]

    # 按步长进行节点离散化降采样，对应模型中的候选节点集 V 构建
    df_sampled = df_route.iloc[::sample_step].copy().reset_index(drop=True)
    coords = df_sampled[[col_x, col_y, col_z]].values

    # 读取附件3县界数据用于背景渲染
    xls_bounds = pd.ExcelFile(boundary_path)
    county_boundaries = {}
    for sheet in xls_bounds.sheet_names:
        df_county = pd.read_excel(boundary_path, sheet_name=sheet)
        county_boundaries[sheet] = df_county.values

    return coords, county_boundaries, df_sampled


def build_viewshed_topology_graph(coords, d_max, z_offset):
    # 初始化无向无权拓扑图 G = (V, E)
    G = nx.Graph()
    n_nodes = len(coords)
    G.add_nodes_from(range(n_nodes))

    # 构建视域连通布尔算子
    for i in range(n_nodes):
        xi, yi, zi = coords[i]
        zi_adjusted = zi + z_offset

        for j in range(i + 1, n_nodes):
            xj, yj, zj = coords[j]
            zj_adjusted = zj + z_offset

            # 计算水平欧氏距离
            dist_ij = math.hypot(xj - xi, yj - yi)

            # 若超出光学最大物理极值距离，直接截断
            if dist_ij > d_max:
                continue

            # 执行视线干涉检验，扫描 L_{i,j} 之间的所有中间节点
            is_visible = True
            for k in range(i + 1, j):
                xk, yk, zk = coords[k]
                dist_ik = math.hypot(xk - xi, yk - yi)

                # 计算比例系数 t
                t = dist_ik / dist_ij

                h_sight = zi_adjusted + t * (zj_adjusted - zi_adjusted)

                if h_sight <= zk:
                    is_visible = False
                    break

            if is_visible:
                G.add_edge(i, j)

    return G


def plot_topology_network(coords, G, shortest_path):
    # 绘制：新秦直道视域拓扑连通网络图
    plt.figure(figsize=(10, 8))
    x, y = coords[:, 0], coords[:, 1]

    # 渲染全量候选节点
    plt.scatter(x, y, c='gray', s=5, alpha=0.5, label='候选节点集 V')

    # 渲染视域拓扑连通边集 E
    for (u, v) in G.edges():
        plt.plot([x[u], x[v]], [y[u], y[v]], c='lightblue', lw=0.2, alpha=0.3)

    # 高亮渲染最优最小跳数路径
    path_x = [x[idx] for idx in shortest_path]
    path_y = [y[idx] for idx in shortest_path]
    plt.plot(path_x, path_y, c='red', lw=2, label='最优通信主干链路')
    plt.scatter(path_x, path_y, c='darkred', marker='^', s=40, label='重规划烽火台', zorder=5)

    plt.title('新秦直道视域拓扑连通网络图')
    plt.xlabel('X 坐标')
    plt.ylabel('Y 坐标')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_line_of_sight_profile(coords, idx_start, idx_end, z_offset):
    # 绘制：烽火台视线纵剖面叠加图
    plt.figure(figsize=(10, 4))

    # 提取两塔之间的真实地形高程序列
    segment_coords = coords[idx_start:idx_end + 1]
    distances = [0]
    for k in range(1, len(segment_coords)):
        dx = segment_coords[k, 0] - segment_coords[k - 1, 0]
        dy = segment_coords[k, 1] - segment_coords[k - 1, 1]
        distances.append(distances[-1] + math.hypot(dx, dy))

    elevations = segment_coords[:, 2]

    # 渲染地形实体剖面
    plt.plot(distances, elevations, c='#8B4513', label='地形实体剖面', lw=1.5)
    plt.fill_between(distances, min(elevations) - 50, elevations, color='#D2B48C', alpha=0.5)

    # 渲染光学直射视线
    h_start = elevations[0] + z_offset
    h_end = elevations[-1] + z_offset
    plt.plot([distances[0], distances[-1]], [h_start, h_end], c='red', linestyle='--', lw=2,
             label='光学直射链路')

    plt.title(f'烽火台视线纵剖面叠加图 (节点 {idx_start} 至节点 {idx_end})')
    plt.xlabel('空间演化里程')
    plt.ylabel('绝对高程')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_overall_map(coords, shortest_path, county_boundaries):
    # 绘制：甘陕八县新秦直道与烽火台全局分布底图
    plt.figure(figsize=(10, 10))

    # 循环渲染八县地理多边形边界约束
    colors = plt.cm.Set3(np.linspace(0, 1, len(county_boundaries)))
    for idx, (county_name, bounds) in enumerate(county_boundaries.items()):
        plt.fill(bounds[:, 0], bounds[:, 1], color=colors[idx], alpha=0.4, edgecolor='gray',
                 label=county_name if idx < 8 else "")
        # 在多边形质心标注县名
        cx, cy = np.mean(bounds[:, 0]), np.mean(bounds[:, 1])
        plt.text(cx, cy, county_name, ha='center', va='center', fontsize=9)

    # 渲染新路线主干道
    plt.plot(coords[:, 0], coords[:, 1], c='darkred', lw=2.5, label='新秦直道重构干线')

    # 渲染烽火台最优离散落点
    beacon_coords = coords[shortest_path]
    plt.scatter(beacon_coords[:, 0], beacon_coords[:, 1], c='yellow', edgecolors='black', marker='*', s=150,
                label='烽火台', zorder=10)

    plt.title('甘陕八县空间控制网及历史军事遗迹耦合分布图')
    plt.xlabel('经向投影坐标 X / m')
    plt.ylabel('纬向投影坐标 Y / m')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 定义绝对数据源路径
    ROUTE_PATH = r"C:\Users\jack\Desktop\编写2026年泰迪杯\佐证材料\新秦直道的路线坐标.xlsx"
    BOUNDARY_PATH = r"C:\Users\jack\Desktop\2026年泰迪杯赛题与优秀论文\A题\全部数据\正式数据\附件3  甘陕八县的县界数据.xlsx"

    # 超参数声明
    SAMPLE_STEP = 50
    D_MAX = 30000
    Z_OFFSET = 10

    print("模型初始化：正在加载数据并执行空间连续流形离散化")
    coords, county_boundaries, df_sampled = load_and_preprocess_data(ROUTE_PATH, BOUNDARY_PATH, SAMPLE_STEP)
    print(f"数据加载完毕，当前空间特征提取节点总数：{len(coords)}")

    print("数学图论映射：正在构建高维视域拓扑邻接矩阵，请稍候")
    G = build_viewshed_topology_graph(coords, D_MAX, Z_OFFSET)

    print("优化求解：执行广度优先目标函数求解最小支配集")
    # 利用 BFS 算法在无权图中求解绝对极小跳数
    source_node = 0
    target_node = len(coords) - 1

    try:
        shortest_path = nx.shortest_path(G, source=source_node, target=target_node)
        print("\n=核心求解结果")
        print(f"全局最少烽火台布设数量极值：{len(shortest_path)} 座")
        print(f"提取的最优空间节点序号序列：{shortest_path}")

        print("生成空间拓扑网络图")
        plot_topology_network(coords, G, shortest_path)

        print("生成微观物理视线校验剖面图")
        # 选取路径中的前两个节点进行局部剖析映射
        if len(shortest_path) >= 2:
            plot_line_of_sight_profile(coords, shortest_path[0], shortest_path[1], Z_OFFSET)

        print("生成地理拓扑边界底图耦合可视化")
        plot_overall_map(coords, shortest_path, county_boundaries)

    except nx.NetworkXNoPath:
        print("当前约束下不存在全局可行解")