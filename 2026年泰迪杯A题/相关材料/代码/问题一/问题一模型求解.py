import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from sklearn.neighbors import KDTree
import gc
import os

# 配置中文字体与负号显示。
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def process_qinzhidao_features(dem_file, att2_file):
    print("1. 正在读取秦直道路线数据，计算研究区域边界。")
    # 提取轨迹点坐标。
    df_route = pd.read_excel(att2_file, sheet_name='秦直道')
    route_x = df_route['x坐标/m'].values
    route_y = df_route['y坐标/m'].values

    # 计算路线包围盒，并外扩 7000 米作为缓冲带。
    buffer = 7000.0
    min_x, max_x = route_x.min() - buffer, route_x.max() + buffer
    min_y, max_y = route_y.min() - buffer, route_y.max() + buffer

    print("2. 正在读取全局高程矩阵并进行局部裁剪。")
    # 读取高程数据。
    df_dem = pd.read_csv(dem_file, index_col=0)
    X_coords_full = df_dem.columns.astype(float).values
    Y_coords_full = df_dem.index.astype(float).values
    Z_raw_full = df_dem.values.astype(float)

    print(f"原始矩阵大小：{Z_raw_full.shape}。")

    # 生成裁剪掩膜，只保留缓冲带内的数据。
    mask_x = (X_coords_full >= min_x) & (X_coords_full <= max_x)
    mask_y = (Y_coords_full >= min_y) & (Y_coords_full <= max_y)

    # 提取裁剪后的局部矩阵。
    X_coords = X_coords_full[mask_x]
    Y_coords = Y_coords_full[mask_y]
    Z_raw = Z_raw_full[mask_y, :][:, mask_x]

    print(f"裁剪后局部矩阵大小：{Z_raw.shape}。")

    # 释放不需要的大矩阵内存。
    del df_dem, X_coords_full, Y_coords_full, Z_raw_full
    gc.collect()

    print("3. 正在执行局部插值与样条曲面重构。")
    # 对坐标进行升序排列以满足插值算法要求。
    x_idx = np.argsort(X_coords)
    y_idx = np.argsort(Y_coords)
    X_sorted = X_coords[x_idx]
    Y_sorted = Y_coords[y_idx]
    Z_sorted = Z_raw[y_idx, :][:, x_idx]

    # 填补局部矩阵中的空值。
    df_z = pd.DataFrame(Z_sorted)
    Z_filled = df_z.interpolate(method='linear', axis=1, limit_direction='both') \
        .interpolate(method='linear', axis=0, limit_direction='both').values

    # 构建双三次样条曲面。
    spline = RectBivariateSpline(Y_sorted, X_sorted, Z_filled, kx=3, ky=3)

    print("4. 正在构建水文遗迹空间搜索树。")
    # 提取河网与分水岭坐标。
    df_river = pd.read_excel(att2_file, sheet_name='河网')
    df_watershed = pd.read_excel(att2_file, sheet_name='一级分水岭')
    river_pts = df_river[['x坐标/m', 'y坐标/m']].values
    watershed_pts = df_watershed[['x坐标/m', 'y坐标/m']].values

    # 构建 KD 树结构用于计算最短距离。
    tree_river = KDTree(river_pts, leaf_size=30, metric='euclidean')
    tree_watershed = KDTree(watershed_pts, leaf_size=30, metric='euclidean')

    print("5. 正在执行向量化批量特征提取。")
    # 批量获取解析高程与连续坡度
    elevs = spline.ev(route_y, route_x)
    dz_dx = spline.ev(route_y, route_x, dx=0, dy=1)
    dz_dy = spline.ev(route_y, route_x, dx=1, dy=0)
    slopes = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    # 批量计算地形位置指数 (TPI) 与起伏度 (TRI)
    step = 30.0
    dx_arr = np.array([-step, 0, step, -step, step, -step, 0, step])
    dy_arr = np.array([-step, -step, -step, 0, 0, step, step, step])

    # 矩阵广播：一次性计算所有点的 8 个邻居坐标
    route_x_mesh = route_x[:, None] + dx_arr[None, :]
    route_y_mesh = route_y[:, None] + dy_arr[None, :]
    neighbors_z_flat = spline.ev(route_y_mesh.flatten(), route_x_mesh.flatten())
    neighbors_z = neighbors_z_flat.reshape(len(route_x), 8)

    tpi = elevs - np.mean(neighbors_z, axis=1)
    tri = np.sqrt(np.sum((neighbors_z - elevs[:, None]) ** 2, axis=1) / 8.0)

    # 批量查询距河网和分水岭的最小距离
    route_pts = np.column_stack((route_x, route_y))
    dist_river, _ = tree_river.query(route_pts, k=1)
    dist_watershed, _ = tree_watershed.query(route_pts, k=1)

    # 批量计算累积里程和离散坡度
    dists = np.sqrt(np.diff(route_x) ** 2 + np.diff(route_y) ** 2)
    cumulative_mileage = np.insert(np.cumsum(dists), 0, 0.0)

    dz_discrete = np.diff(elevs)
    discrete_slopes = np.insert(np.degrees(np.arctan(np.abs(dz_discrete) / np.maximum(dists, 1e-6))), 0, 0.0)

    print("6. 特征提取完毕，正在绘制对比图与保存结果。")
    # 保存特征数据到 Excel 。
    cols = ['序号', 'x坐标/m', 'y坐标/m', '高程', '连续坡度', '地形位置指数', '地形起伏度', '距最近河网', '距最近一级分水岭']
    df_res = pd.DataFrame({
        '序号': np.arange(1, len(route_x) + 1),
        'x坐标/m': route_x,
        'y坐标/m': route_y,
        '高程': elevs,
        '连续坡度': slopes,
        '地形位置指数': tpi,
        '地形起伏度': tri,
        '距最近河网': dist_river.flatten(),
        '距最近一级分水岭': dist_watershed.flatten()
    }, columns=cols)
    df_res.to_excel('result1.xlsx', index=False)

    # 绘制离散坡度图。
    fig1, ax1 = plt.subplots(figsize=(10, 4), dpi=300)
    ax1.plot(cumulative_mileage, discrete_slopes, color='#D32F2F', linewidth=1.2, alpha=0.8)
    ax1.set_title("最近邻离散差分坡度", fontsize=12)
    ax1.set_xlabel("里程 / m")
    ax1.set_ylabel("离散坡度 / °")
    plt.tight_layout()
    plt.savefig('Plot1_Discrete_Slope.png', dpi=300)
    plt.close()

    # 绘制平滑坡度图。
    fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=300)
    ax2.plot(cumulative_mileage, slopes, color='#1976D2', linewidth=1.5)
    ax2.set_title("双三次样条解析坡度", fontsize=12)
    ax2.set_xlabel("里程 / m")
    ax2.set_ylabel("连续坡度 / °")
    plt.tight_layout()
    plt.savefig('Plot2_Continuous_Spline_Slope.png', dpi=300)
    plt.close()

    # 绘制地形特征演化图。
    fig3, (ax3, ax4, ax5) = plt.subplots(3, 1, figsize=(12, 10), dpi=300, sharex=True)

    ax3.plot(cumulative_mileage, elevs, color='#388E3C', linewidth=1.5)
    ax3.set_ylabel("绝对高程 / m")
    ax3.set_title("秦直道多维地形特征空间演化序列图", fontsize=14, pad=15)

    ax4.plot(cumulative_mileage, tpi, color='#F57C00', linewidth=1.5)
    ax4.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax4.set_ylabel("TPI 位置指数")

    ax5.plot(cumulative_mileage, dist_river.flatten(), color='#0288D1', linewidth=1.5)
    ax5.plot(cumulative_mileage, dist_watershed.flatten(), color='#7B1FA2', linewidth=1.5, alpha=0.7)
    ax5.legend(["距河网拓扑距离", "距分水岭拓扑距离"], loc='upper right')
    ax5.set_ylabel("空间阻力距离 / m")
    ax5.set_xlabel("路线演化总里程 / m")

    plt.tight_layout()
    plt.savefig('Plot3_Feature_Evolution_Sequence.png', dpi=300)
    plt.close()

    # 打印完成提示。
    print(f"解析成功，特征集行数：{len(df_res)}，提取结果已保存至 result1.xlsx。")


if __name__ == '__main__':
    # 设置文件路径并执行。
    DEM_PATH = r"C:\Users\jack\Desktop\编写2026年泰迪杯\佐证材料\陕甘八县的高程数据_清洗后.csv"
    ATT2_PATH = r"C:\Users\jack\Desktop\2026年泰迪杯赛题与优秀论文\A题\全部数据\正式数据\附件2  秦直道及周边地形和相关遗迹的数据.xlsx"

    process_qinzhidao_features(DEM_PATH, ATT2_PATH)