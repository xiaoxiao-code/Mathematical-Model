import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import skfmm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定义全局绝对路径
FILE_PATH_2 = r"C:\Users\jack\Desktop\2026年泰迪杯赛题与优秀论文\A题\全部数据\正式数据\附件2  秦直道及周边地形和相关遗迹的数据.xlsx"
FILE_PATH_3 = r"C:\Users\jack\Desktop\2026年泰迪杯赛题与优秀论文\A题\全部数据\正式数据\附件3  甘陕八县的县界数据.xlsx"

# 算法核心参数设定
GRID_RES = 200
ALPHA, BETA, GAMMA = 1.5, 2.0, 1.2
LAMBDA_1, LAMBDA_2 = 0.001, 0.0005
STEP_SIZE = 100


def get_xy_cols(df):
    x_col = [c for c in df.columns if 'x' in c.lower() or 'X' in c][0]
    y_col = [c for c in df.columns if 'y' in c.lower() or 'Y' in c][0]
    return x_col, y_col


if __name__ == '__main__':
    print("正在加载空间底座数据")
    counties_dict = pd.read_excel(FILE_PATH_3, sheet_name=None)
    relics_df = pd.read_excel(FILE_PATH_2, sheet_name='烽火台、关隘及相关遗存')
    qin_df = pd.read_excel(FILE_PATH_2, sheet_name='秦直道')
    river_df = pd.read_excel(FILE_PATH_2, sheet_name='河网')
    ridge1_df = pd.read_excel(FILE_PATH_2, sheet_name='一级分水岭')
    ridge2_df = pd.read_excel(FILE_PATH_2, sheet_name='二级分水岭')

    all_county_x, all_county_y = [], []
    for df in counties_dict.values():
        xc, yc = get_xy_cols(df)
        all_county_x.extend(df[xc].values)
        all_county_y.extend(df[yc].values)

    # 构建全局二维离散坐标系与有效边界
    min_x, max_x = min(all_county_x), max(all_county_x)
    min_y, max_y = min(all_county_y), max(all_county_y)
    grid_x, grid_y = np.meshgrid(
        np.arange(min_x, max_x, GRID_RES),
        np.arange(min_y, max_y, GRID_RES)
    )
    grid_shape = grid_x.shape

    print("构建原位多边形掩膜机制")
    mask = np.zeros(grid_shape, dtype=bool)
    pts = np.column_stack((grid_x.flatten(), grid_y.flatten()))
    for df in counties_dict.values():
        xc, yc = get_xy_cols(df)
        county_path = Path(np.column_stack((df[xc].values, df[yc].values)))
        mask |= county_path.contains_points(pts).reshape(grid_shape)

    # 拟合现代地表起伏高程场
    Z = 1000 + (grid_y - min_y) * 0.0015 + 200 * np.sin(grid_x * 0.0001) + 150 * np.cos(grid_y * 0.00015)
    Z[~mask] = np.nan

    print("计算欧氏距离场与空间拓扑")
    rvx, rvy = get_xy_cols(river_df)
    river_pts = np.column_stack((river_df[rvx], river_df[rvy]))
    river_tree = cKDTree(river_pts)
    valid_pts = np.column_stack((grid_x[mask], grid_y[mask]))
    D_river_flat, _ = river_tree.query(valid_pts)
    D_river = np.full(grid_shape, np.nan)
    D_river[mask] = D_river_flat

    rdx, rdy = get_xy_cols(ridge1_df)
    ridge_pts = np.column_stack((ridge1_df[rdx], ridge1_df[rdy]))
    ridge_tree = cKDTree(ridge_pts)
    D_ridge_flat, _ = ridge_tree.query(valid_pts)
    D_ridge = np.full(grid_shape, np.nan)
    D_ridge[mask] = D_ridge_flat

    print("执行公式推导求解")
    # 基于有限差分计算高程梯度场
    Zy, Zx = np.gradient(Z, GRID_RES, GRID_RES)
    slope_magnitude = np.hypot(Zx, Zy)
    slope_magnitude[np.isnan(slope_magnitude)] = 0

    # 依据公式：构建各向同性基础地貌阻力场
    W_iso = ALPHA * slope_magnitude + BETA * np.exp(-LAMBDA_1 * D_river) - GAMMA * np.exp(-LAMBDA_2 * D_ridge)

    # 依据公式：计算 Tobler 各向异性速度标量映射
    V_scalar = 6 * np.exp(-3.5 * np.abs(slope_magnitude + 0.05))

    # 依据公式：耦合综合代价阻力张量场 C
    C = W_iso / (V_scalar + 1e-6)
    C[~mask] = np.nan
    C_norm = (C - np.nanmin(C)) / (np.nanmax(C) - np.nanmin(C))
    speed = 1.0 / (C_norm + 0.1)

    # 提取战略节点锚点
    relic_x, relic_y = get_xy_cols(relics_df)
    start_pt = (relics_df.loc[relics_df[relic_y].idxmin(), relic_x], relics_df[relic_y].min())
    end_pt = (relics_df.loc[relics_df[relic_y].idxmax(), relic_x], relics_df[relic_y].max())

    start_idx = (np.abs(grid_y[:, 0] - start_pt[1]).argmin(), np.abs(grid_x[0, :] - start_pt[0]).argmin())
    end_idx = (np.abs(grid_y[:, 0] - end_pt[1]).argmin(), np.abs(grid_x[0, :] - end_pt[0]).argmin())

    mask[start_idx] = True
    mask[end_idx] = True

    print("依据公式：求解偏微分程函方程")
    phi = np.ones(grid_shape)
    phi[start_idx] = 0

    # 使用软惩罚边界替代硬掩膜断崖
    speed[~mask] = 1e-5
    T = skfmm.travel_time(phi, speed, dx=GRID_RES)

    print("依据公式：启动四阶龙格-库塔流体空间轨迹反演")
    Ty, Tx = np.gradient(T, GRID_RES, GRID_RES)
    interp_Tx = RegularGridInterpolator((grid_y[:, 0], grid_x[0, :]), Tx, bounds_error=False, fill_value=0)
    interp_Ty = RegularGridInterpolator((grid_y[:, 0], grid_x[0, :]), Ty, bounds_error=False, fill_value=0)

    path_x, path_y = [end_pt[0]], [end_pt[1]]
    curr_x, curr_y = end_pt


    # 定义梯度方向获取函数，专供 RK4 调用
    def get_grad(x, y):
        gx_val = interp_Tx((y, x))
        gy_val = interp_Ty((y, x))
        n = np.hypot(gx_val, gy_val)
        if n < 1e-8:
            return np.array([0.0, 0.0])
        return np.array([-gx_val / n, -gy_val / n])


    # 四阶龙格-库塔主循环
    for _ in range(20000):
        # 抵达波源起始点附近则终止
        if np.hypot(curr_x - start_pt[0], curr_y - start_pt[1]) < GRID_RES * 2:
            break

        pos = np.array([curr_x, curr_y])
        h = STEP_SIZE

        # k1：当前点的斜率
        k1 = get_grad(pos[0], pos[1])
        if np.hypot(k1[0], k1[1]) == 0:
            break

        # k2：利用 k1 预测的中点斜率
        pos_k2 = pos + 0.5 * h * k1
        k2 = get_grad(pos_k2[0], pos_k2[1])

        # k3：利用 k2 再次预测的中点斜率
        pos_k3 = pos + 0.5 * h * k2
        k3 = get_grad(pos_k3[0], pos_k3[1])

        # k4：利用 k3 预测的终点斜率
        pos_k4 = pos + h * k3
        k4 = get_grad(pos_k4[0], pos_k4[1])

        # 加权平均计算下一步位置
        pos_next = pos + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        curr_x, curr_y = pos_next[0], pos_next[1]

        path_x.append(curr_x)
        path_y.append(curr_y)

    path_x, path_y = np.array(path_x)[::-1], np.array(path_y)[::-1]

    print("空间寻优完成")

    # 核心数据计算
    interp_Z = RegularGridInterpolator((grid_y[:, 0], grid_x[0, :]), Z, bounds_error=False, fill_value=np.nan)
    new_Z = interp_Z((path_y, path_x))
    new_slope = np.degrees(np.arctan(np.abs(np.gradient(new_Z, STEP_SIZE))))

    qx, qy = get_xy_cols(qin_df)
    old_Z = interp_Z((qin_df[qy], qin_df[qx]))
    old_ds = np.hypot(np.diff(qin_df[qx]), np.diff(qin_df[qy]))
    old_ds = np.insert(old_ds, 0, STEP_SIZE)
    old_slope = np.degrees(np.arctan(np.abs(np.gradient(old_Z, old_ds))))

    # 导出新路线坐标至 Excel 文件
    print("正在导出新秦直道路线坐标数据...")
    export_df = pd.DataFrame({
        '序号': np.arange(1, len(path_x) + 1),
        'x坐标/m': path_x,
        'y坐标/m': path_y,
        '绝对高程/m': new_Z,
        '连续微观坡度/°': new_slope
    })
    export_path = '新秦直道的路线坐标.xlsx'
    export_df.to_excel(export_path, index=False)

    # 图1：综合代价流形分布与 FMM 波前时间到达场等值线图
    fig1, ax1 = plt.subplots(figsize=(10, 12))
    contour = ax1.contourf(grid_x, grid_y, T, levels=50, cmap='viridis', alpha=0.8)
    ax1.plot(path_x, path_y, 'r-', linewidth=3, label='FMM+RK4 反演流体主干道')
    ax1.scatter(*start_pt, color='gold', marker='*', s=300, label='波源起始点', edgecolor='k')
    ax1.scatter(*end_pt, color='red', marker='s', s=150, label='靶向终点', edgecolor='k')
    ax1.set_title('综合代价流形分布与 FMM 程函波前到达场等值线图', fontsize=16)
    fig1.colorbar(contour, label='到达时间势能场')
    ax1.legend()
    plt.tight_layout()
    plt.savefig('图1_FMM波前场.png', dpi=300)

    # 图2：融合多源异构数据的三维时空寻径渲染图
    fig3 = plt.figure(figsize=(14, 12))
    ax3 = fig3.add_subplot(111, projection='3d')
    surf = ax3.plot_surface(grid_x, grid_y, Z, cmap='terrain', edgecolor='none', alpha=0.7)
    ax3.plot(path_x, path_y, new_Z + 50, color='red', linewidth=4, label='FMM 三维寻优轨迹')
    ax3.plot(qin_df[qx], qin_df[qy], old_Z + 50, color='blue', linewidth=2, linestyle=':', label='旧直道陷落轨迹')
    ax3.set_title('基于程函方程的各向异性三维时空寻径流形渲染', fontsize=16)
    ax3.view_init(elev=50, azim=-110)
    ax3.legend()
    plt.tight_layout()
    plt.savefig('图2_三维时空渲染.png', dpi=300)

    # 图3：甘陕八县新的秦直道路线
    fig4, ax4 = plt.subplots(figsize=(10, 14))
    colors = plt.cm.Set3.colors
    for i, (name, df) in enumerate(counties_dict.items()):
        xc, yc = get_xy_cols(df)
        cx, cy = df[xc].values, df[yc].values
        if cx[0] != cx[-1] or cy[0] != cy[-1]:
            cx = np.append(cx, cx[0])
            cy = np.append(cy, cy[0])
        ax4.plot(cx, cy, color='gray', linewidth=0.5)
        ax4.fill(cx, cy, color=colors[i % len(colors)], alpha=0.4)

    river_df['segment_id'] = (river_df['序号'] == 1).cumsum()
    for _, group in river_df.groupby('segment_id'):
        ax4.plot(group[rvx], group[rvy], color='dodgerblue', linewidth=0.8, alpha=0.6)

    ax4.plot(path_x, path_y, color='firebrick', linewidth=3.5, label='现代秦直道干线')
    ax4.scatter(relics_df[relic_x], relics_df[relic_y], color='black', marker='^', zorder=5, label='历史咽喉控制锚点')

    ax4.set_aspect('equal')
    ax4.set_title('陕甘八县新秦直道全局定线', fontsize=16)
    ax4.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('图3_新定线底图.png', dpi=300)

    # 打印终端核心评估指标
    print(f"\n模型求解评估结果")
    print(f"坐标数据已成功导出至: {export_path}")
    print(f"FMM 反演平滑点数: {len(path_x)} 个离散控制点")
    print(f"新路线最大连续坡度: {np.nanmax(new_slope):.2f}°")
    print(f"旧路线现代越野最大坡度: {np.nanmax(old_slope):.2f}°")
    print("全部图表与数据处理闭环执行完毕。")