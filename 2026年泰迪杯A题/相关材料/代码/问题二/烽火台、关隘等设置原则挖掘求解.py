import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.ndimage import laplace, maximum_filter, minimum_filter
from scipy.spatial import cKDTree  # 新增：用于计算空间距离掩膜
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def bresenham_line(x0, y0, x1, y1):
    # 依据光线直线传播原理，提取三维视线路径上的二维离散栅格坐标序列
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def check_los(grid_H, cx, cy, tx, ty, h1, h2, cell_size):
    # 将倒置的 [cy, cx] 修改为正确的 [cx, cy]
    H_o = grid_H[cx, cy]
    H_t = grid_H[tx, ty]
    d_OT = np.hypot(tx - cx, ty - cy) * cell_size
    if d_OT == 0:
        return 1
    tan_theta_OT = ((H_t + h2) - (H_o + h1)) / d_OT

    # 获取视线路径上的所有中间栅格点
    line_points = bresenham_line(cx, cy, tx, ty)

    # 遍历中间节点，应用公式进行空间遮挡判定
    for px, py in line_points[1:-1]:
        # 将倒置的 [py, px] 修改为正确的 [px, py]
        H_p = grid_H[px, py]
        d_OP = np.hypot(px - cx, py - cy) * cell_size
        tan_theta_OP = (H_p - (H_o + h1)) / d_OP
        if tan_theta_OP > tan_theta_OT:
            return 0
    return 1


if __name__ == '__main__':
    result1_path = r"C:\Users\jack\Desktop\编写2026年泰迪杯\佐证材料\result1.xlsx"
    relics_path = r"C:\Users\jack\Desktop\2026年泰迪杯赛题与优秀论文\A题\全部数据\正式数据\附件2  秦直道及周边地形和相关遗迹的数据.xlsx"

    # 载入问题一输出的全量地理坐标与地形特征矩阵
    df_result1 = pd.read_excel(result1_path)
    X_coords = df_result1['x坐标/m'].values
    Y_coords = df_result1['y坐标/m'].values
    H_vals = df_result1['高程'].values
    S_vals = df_result1['连续坡度'].values

    # 载入遗迹点数据并进行类型匹配过滤
    df_relics = pd.read_excel(relics_path, sheet_name='烽火台、关隘及相关遗存')
    df_beacons = df_relics[df_relics['类型'].str.contains('烽火台')]
    df_passes = df_relics[df_relics['类型'].str.contains('关隘')]

    # 设定基准空间物理参数
    grid_res = 200
    h1 = 10.0
    h2 = 2.0
    R_limit = 15000
    R_cells = int(R_limit / grid_res)
    alpha = 1.0
    beta = 0.05
    gamma = 0.5

    # 构建二维连续拓扑光栅矩阵域
    x_min, x_max = X_coords.min(), X_coords.max()
    y_min, y_max = Y_coords.min(), Y_coords.max()
    grid_x, grid_y = np.mgrid[x_min:x_max:grid_res, y_min:y_max:grid_res]

    # 通过空间插值将离散地形特征重构为高精度二维光栅矩阵
    grid_H = griddata((X_coords, Y_coords), H_vals, (grid_x, grid_y), method='linear')
    grid_S = griddata((X_coords, Y_coords), S_vals, (grid_x, grid_y), method='linear')
    grid_H = np.nan_to_num(grid_H, nan=np.nanmean(grid_H))
    grid_S = np.nan_to_num(grid_S, nan=0)

    # 构建目标函数：计算考虑坡度与二阶起伏的各向异性空间基础通行阻力表面
    laplacian_H = laplace(grid_H)
    C_base = alpha * np.exp(beta * grid_S) + gamma * np.abs(laplacian_H)

    # 应用公式：提取局部成本汇构成咽喉约束方程
    C_max = maximum_filter(C_base, size=3)
    C_min = minimum_filter(C_base, size=3)
    Choke_index = C_max / (C_min + 1e-5)

    beacon_cv_scores = []
    global_viewshed = np.zeros_like(grid_H)


    # 将遗迹点坐标映射至降维后的栅格索引空间
    def coord_to_idx(x, y):
        idx_x = int(np.clip((x - x_min) / grid_res, 0, grid_x.shape[0] - 1))
        idx_y = int(np.clip((y - y_min) / grid_res, 0, grid_x.shape[1] - 1))
        return idx_x, idx_y


    print("开始计算烽火台视域网络拓扑")
    for _, row in df_beacons.iterrows():
        bx, by = coord_to_idx(row['x坐标/m'], row['y坐标/m'])
        local_cv = 0

        # 理顺边界定义的顺序
        x_lim, y_lim = grid_H.shape

        # 降采样扫描局部视域，应用公式计算局部累积视域覆盖度
        for dy in range(-R_cells, R_cells + 1, 5):
            for dx in range(-R_cells, R_cells + 1, 5):
                if dx ** 2 + dy ** 2 <= R_cells ** 2:
                    tx, ty = bx + dx, by + dy
                    # 边界判断使用理顺后的极限值
                    if 0 <= tx < x_lim and 0 <= ty < y_lim:
                        visible = check_los(grid_H, bx, by, tx, ty, h1, h2, grid_res)
                        local_cv += visible
                        global_viewshed[tx, ty] = max(global_viewshed[tx, ty], visible)
        beacon_cv_scores.append(local_cv)

    # 基于反事实假设，生成同等数量的随机背景点计算以进行对比证明
    random_cv_scores = []
    for _ in range(len(df_beacons)):
        rx, ry = np.random.randint(0, grid_H.shape[0]), np.random.randint(0, grid_H.shape[1])
        local_cv = 0
        for dy in range(-R_cells, R_cells + 1, 5):
            for dx in range(-R_cells, R_cells + 1, 5):
                if dx ** 2 + dy ** 2 <= R_cells ** 2:
                    tx, ty = rx + dx, ry + dy
                    if 0 <= tx < grid_H.shape[0] and 0 <= ty < grid_H.shape[1]:
                        local_cv += check_los(grid_H, rx, ry, tx, ty, h1, h2, grid_res)
        random_cv_scores.append(local_cv)

    print(f"烽火台平均可视域得分: {np.mean(beacon_cv_scores):.2f}")
    print(f"随机点平均可视域得分: {np.mean(random_cv_scores):.2f}")

    # 构建空间距离掩膜
    # 将远离真实道路的无意义插值区域切除，保留带状走廊
    tree = cKDTree(np.c_[X_coords, Y_coords])
    grid_coords = np.c_[grid_x.ravel(), grid_y.ravel()]
    distances, _ = tree.query(grid_coords)
    distances = distances.reshape(grid_x.shape)

    # 设定有效缓冲半径
    mask = distances > 18000

    # 应用掩膜：将背景置为 NaN
    grid_H_masked = np.where(mask, np.nan, grid_H)
    Choke_index_masked = np.where(mask, np.nan, Choke_index)

    # 视域覆盖图中，将没有视域的地方也完全透明化
    global_viewshed_masked = np.where(mask, np.nan, global_viewshed)
    global_viewshed_masked = np.where(global_viewshed_masked == 0, np.nan, global_viewshed_masked)

    # 全局绘图样式设置
    plt.rcParams['axes.facecolor'] = '#F8F9FA'
    plt.rcParams['figure.facecolor'] = '#FFFFFF'

    # 烽火台三维累积视域覆盖热力图
    plt.figure(figsize=(10, 12), dpi=300)

    # 绘制带状地形底图
    plt.imshow(grid_H_masked.T, cmap='terrain', origin='lower', alpha=0.35)

    # 叠加视域热力图
    plt.imshow(global_viewshed_masked.T, cmap='YlOrRd', origin='lower', alpha=0.85)

    # 绘制烽火台节点
    bx_list = [coord_to_idx(r['x坐标/m'], r['y坐标/m'])[0] for _, r in df_beacons.iterrows()]
    by_list = [coord_to_idx(r['x坐标/m'], r['y坐标/m'])[1] for _, r in df_beacons.iterrows()]
    plt.scatter(bx_list, by_list, c='#D32F2F', marker='*', s=180,
                edgecolors='white', linewidths=1.2, zorder=5, label='战术烽火台')

    plt.title('秦直道烽火台累积视域光学预警网络', fontsize=16, fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.axis('off')  # 关闭坐标轴边框
    plt.tight_layout()
    plt.savefig('Viewshed_Coverage_Heatmap_Pro.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    # 关隘地形咽喉指数叠置场
    plt.figure(figsize=(10, 12), dpi=300)

    # 绘制咽喉成本汇场
    vmax_val = np.percentile(Choke_index_masked[~np.isnan(Choke_index_masked)], 92)
    plt.imshow(Choke_index_masked.T, cmap='magma', origin='lower', vmax=vmax_val, alpha=0.9)

    # 绘制关隘节点
    px_list = [coord_to_idx(r['x坐标/m'], r['y坐标/m'])[0] for _, r in df_passes.iterrows()]
    py_list = [coord_to_idx(r['x坐标/m'], r['y坐标/m'])[1] for _, r in df_passes.iterrows()]
    plt.scatter(px_list, py_list, c='#FFC107', marker='^', s=200,
                edgecolors='white', linewidths=1.5, zorder=5, label='核心关隘防御节点')

    plt.title('关隘军事拓扑成本汇与地形咽喉指数场', fontsize=16, fontweight='bold', pad=20)
    plt.legend(frameon=True, fancybox=True, shadow=True, borderpad=1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('Chokepoint_Index_Map_Pro.png', dpi=300, bbox_inches='tight', transparent=False)
    plt.close()

    # 小提琴图显著性检验
    plt.figure(figsize=(8, 6), dpi=300)
    plot_data = pd.DataFrame({
        '视域覆盖度': beacon_cv_scores + random_cv_scores,
        '节点类别': ['历史真实烽火台'] * len(beacon_cv_scores) + ['蒙特卡洛随机点'] * len(random_cv_scores)
    })

    # 使用定制化配色并增加内部细节
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 恢复中文字体
    plt.rcParams['axes.unicode_minus'] = False

    ax = sns.violinplot(x='节点类别', y='视域覆盖度', data=plot_data,
                        palette=['#FF9E80', '#90CAF9'], inner='box', linewidth=1.5)
    # 去除顶部和右侧的边框线
    sns.despine()

    plt.title('烽火台选址累积视域局部极值显著性检验', fontsize=15, fontweight='bold', pad=15)
    plt.ylabel('积分视域覆盖率', fontsize=12)
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig('Viewshed_Optimality_Violin_Pro.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(
        "图表导出完毕：Viewshed_Coverage_Heatmap_Pro.png, Chokepoint_Index_Map_Pro.png, Viewshed_Optimality_Violin_Pro.png")