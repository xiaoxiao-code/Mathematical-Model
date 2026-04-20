import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows默认黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def isolation_forest_outlier_detection(file_path):
    """
    使用多变量类型处理 (孤立森林) 检测高维异常值，并将异常整行高亮标橘色导出。
    """
    # 读取数据
    try:
        df = pd.read_excel(file_path)
        print(f"当前数据规模：{df.shape[0]} 行, {df.shape[1]} 列")
    except Exception as e:
        print(f"请检查路径或文件是否被占用。错误信息：{e}")
        return

    # 选取用于检测的特征列
    # 不能将 '样本ID' 或 '目标标签' 丢入无监督异常检测，否则会产生严重的数据干扰与穿越。
    exclude_cols = ['样本ID', '高血脂症二分类标签', '血脂异常分型标签（确诊病例）']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"剔除标签与ID后，实际参与联合异常检测的特征数：{len(feature_cols)} 个")

    # 构建与训练孤立森林模型
    # 参数说明：
    # - n_estimators=100：构建 100 棵孤立树
    # - contamination=0.05：预期异常值比例，通常医学数据中的严重离群点在 1%~5% 之间，此处设为 5%
    # - random_state=42：固定随机种子，确保论文复现时结果绝对一致
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)

    # 拟合数据并预测
    # 预测结果中，1 代表正常值，-1 代表异常点
    preds = iso_forest.fit_predict(df[feature_cols])

    # 统计异常值
    outlier_mask = (preds == -1)
    outlier_count = outlier_mask.sum()
    total_count = len(df)
    outlier_pct = (outlier_count / total_count) * 100

    print("\n异常值检测结果")
    print(f"发现全局多变量异常值：{outlier_count} 行")
    print(f"异常值在总样本量中的占比：{outlier_pct:.2f}%")

    if outlier_count == 0:
        print("未检测到任何异常数据，无需导出清洗报告")
        return

    # Excel 脏数据高亮渲染规则
    def apply_row_styles(data):
        """
        根据孤立森林预测的掩码，将被判定为异常值的“整行数据”背景色高亮为深橘色。
        """
        # 初始化空样式表
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        # 将异常点对应行的所有列样式置为深橘色
        styles.loc[outlier_mask, :] = 'background-color: darkorange; color: white;'
        return styles

    # 应用样式
    styled_df = df.style.apply(apply_row_styles, axis=None)

    # 构造导出路径
    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    file_name, _ = os.path.splitext(base_name)
    export_path = os.path.join(dir_name, f"{file_name}_多变量检查异常值.xlsx")

    # 导出高亮 Excel
    try:
        styled_df.to_excel(export_path, index=False, engine='openpyxl')
        print(f"已保存至：\n{export_path}")
    except PermissionError:
        print(f"目标文件被占用，请关闭已打开的 {file_name}_多变量检查异常值.xlsx 后重试。")

if __name__ == '__main__':
    # 您指定的绝对路径
    TARGET_FILE_PATH = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据.xlsx"
    isolation_forest_outlier_detection(TARGET_FILE_PATH)