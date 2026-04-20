import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gc

# 配置中文字体与负号显示。
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def process_and_visualize_elevation(file_path, target_3d_size=500):
    # 读取 CSV 文件，将 NA 转为空值，并使用 float32 节省内存。
    df = pd.read_csv(file_path, index_col=0, na_values=['NA'], dtype=np.float32)

    # 提取横纵坐标。
    X_coords = df.columns.astype(float).values
    Y_coords = df.index.astype(float).values

    # 获取高程数据矩阵。
    Z_clean = df.values

    # 释放内存。
    del df
    gc.collect()

    # 生成二值掩膜矩阵，标记有效数据区域。
    mask_matrix = (~np.isnan(Z_clean)).astype(np.int8)

    print(f"矩阵大小：{Z_clean.shape}。")

    # 绘制掩膜二值化分布图。
    fig1, ax1 = plt.subplots(figsize=(8, 8), dpi=300)

    # 设置颜色，深灰色为无效区域，亮黄色为有效区域。
    cmap_binary = ListedColormap(['#363636', '#F5D033'])

    # 对二维绘图进行降采样，防止内存溢出。
    stride_2d_y = max(1, mask_matrix.shape[0] // 2000)
    stride_2d_x = max(1, mask_matrix.shape[1] // 2000)
    mask_plot = mask_matrix[::stride_2d_y, ::stride_2d_x]

    # 绘制分布图。
    cax = ax1.imshow(mask_plot, cmap=cmap_binary,
                     extent=[X_coords.min(), X_coords.max(), Y_coords.min(), Y_coords.max()], origin='upper')

    ax1.set_title("陕甘八县高程矩阵掩膜二值化分布图", fontsize=14, pad=15)
    ax1.set_xlabel("X 坐标 / m", fontsize=12)
    ax1.set_ylabel("Y 坐标 / m", fontsize=12)

    # 添加图例。
    cbar = fig1.colorbar(cax, ticks=[0.25, 0.75], shrink=0.7)
    cbar.ax.set_yticklabels(['境外', '境内'])
    plt.tight_layout()
    plt.savefig('Mask_Binary_Distribution.png', dpi=300, bbox_inches='tight')
    print("二维掩膜分布图已保存。")

    # 绘制处理前后的三维地形对比图。
    # 计算三维渲染的采样步长。
    stride_y = max(1, Z_clean.shape[0] // target_3d_size)
    stride_x = max(1, Z_clean.shape[1] // target_3d_size)

    # 抽样提取数据。
    Z_plot = Z_clean[::stride_y, ::stride_x]
    X_plot_coords = X_coords[::stride_x]
    Y_plot_coords = Y_coords[::stride_y]
    X_mesh, Y_mesh = np.meshgrid(X_plot_coords, Y_plot_coords)

    # 填充空值，模拟处理前的断崖效果。
    min_z = np.nanmin(Z_plot)
    Z_before = np.where(np.isnan(Z_plot), min_z - 200, Z_plot)

    # 绘制处理前的三维图。
    fig2 = plt.figure(figsize=(10, 8), dpi=300)
    ax2 = fig2.add_subplot(1, 1, 1, projection='3d')
    surf1 = ax2.plot_surface(X_mesh, Y_mesh, Z_before, cmap='terrain', rstride=1, cstride=1,
                             linewidth=0, antialiased=True)
    ax2.set_title("异常值干扰导致的空间连续性断裂", fontsize=14, pad=15)
    ax2.set_xlabel('X 坐标 / m')
    ax2.set_ylabel('Y 坐标 / m')
    ax2.set_zlabel('海拔高度 Z / m')

    # 添加颜色条并保存图片。
    fig2.colorbar(surf1, ax=ax2, shrink=0.6, aspect=15, label='海拔高度 (m)', pad=0.1)
    plt.savefig('3D_Elevation_Before.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 绘制处理后的三维图。
    fig3 = plt.figure(figsize=(10, 8), dpi=300)
    ax3 = fig3.add_subplot(1, 1, 1, projection='3d')
    surf2 = ax3.plot_surface(X_mesh, Y_mesh, Z_plot, cmap='terrain', rstride=1, cstride=1,
                             linewidth=0, antialiased=True)
    ax3.set_title("NaN 原位掩膜保留的真实地貌拓扑", fontsize=14, pad=15)
    ax3.set_xlabel('X 坐标 / m')
    ax3.set_ylabel('Y 坐标 / m')
    ax3.set_zlabel('海拔高度 Z / m')

    # 添加颜色条并保存图片。
    fig3.colorbar(surf2, ax=ax3, shrink=0.6, aspect=15, label='海拔高度 (m)', pad=0.1)
    plt.savefig('3D_Elevation_After.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 打印数据处理报告。
    print("\n数据清洗报告：")
    print(f"原矩阵大小：{Z_clean.shape}。")
    print(f"有效数据数量：{np.sum(mask_matrix == 1)}。")
    print(f"隔离空值数量：{np.sum(mask_matrix == 0)}。")

    # 转换为 DataFrame 格式。
    cleaned_df = pd.DataFrame(Z_clean, index=Y_coords, columns=X_coords)

    # 保存为 CSV 文件。
    cleaned_df.to_csv('陕甘八县的高程数据_清洗后.csv', na_rep='NaN')

    # 释放内存。
    del cleaned_df
    gc.collect()

if __name__ == '__main__':
    # 设置文件路径并运行。
    FILE_PATH = r"C:\Users\jack\Desktop\2026年泰迪杯赛题与优秀论文\A题\全部数据\正式数据\附件1  陕甘八县的高程数据\陕甘八县的高程数据.csv"
    process_and_visualize_elevation(file_path=FILE_PATH, target_3d_size=500)