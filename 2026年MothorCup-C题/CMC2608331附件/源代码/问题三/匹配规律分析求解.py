import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 多目标离散动态优化原理构建干预寻优模型
class TCMInterventionProblem(ElementwiseProblem):
    def __init__(self, s_0, l_max):
        self.s_0 = s_0
        self.l_max = l_max
        xl = [1] * 6 + [1] * 6
        xu = [self.l_max] * 6 + [10] * 6
        super().__init__(n_var=12, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        s_t = self.s_0
        total_cost = 0.0

        for t in range(6):
            x_t = int(round(x[t]))
            y_t = int(round(x[t + 6]))

            if s_t <= 58:
                c_tcm = 30
            elif s_t <= 61:
                c_tcm = 80
            else:
                c_tcm = 130

            c_unit_map = {1: 3, 2: 5, 3: 8}
            c_unit = c_unit_map[x_t]
            c_month = c_tcm + 4 * y_t * c_unit
            total_cost += c_month

            if y_t < 5:
                r_t = 0.0
            else:
                r_t = x_t * 0.03 + (y_t - 5) * 0.01

            s_t = s_t * (1 - r_t)
            s_t = max(0.0, min(100.0, s_t))

        out["F"] = [s_t, total_cost]
        out["G"] = [total_cost - 2000.0]


def get_l_max(age_group, score):
    if age_group in [1, 2]:
        l_age = 3
    elif age_group in [3, 4]:
        l_age = 2
    else:
        l_age = 1

    if score < 40:
        l_score = 1
    elif score < 60:
        l_score = 2
    else:
        l_score = 3

    return min(l_age, l_score)


if __name__ == '__main__':
    # 数据
    file_path = r'C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx'

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        try:
            df = pd.read_csv(file_path.replace('.xlsx', ' - Sheet1.csv'))
        except FileNotFoundError:
            print("未找到数据文件，请检查绝对路径。")
            exit()

    df_phlegm = df[df['体质标签'] == 5].copy()

    features = ['年龄组', '活动量表总分（ADL总分+IADL总分）', '痰湿质']
    X_data = df_phlegm[features].copy()
    y_target = []

    # 构建 NSGA-II 算法实体
    algorithm = NSGA2(
        pop_size=50,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.1, vtype=float, repair=RoundingRepair())
    )

    print(f"正在批量生成 {len(df_phlegm)} 个样本的靶向标签，请耐心等待...")

    # 遍历附件所有确诊患者数据，生成对应个体的最优干预时序
    for idx, row in df_phlegm.iterrows():
        s_0 = float(row['痰湿质'])
        age_group = int(row['年龄组'])
        score = float(row['活动量表总分（ADL总分+IADL总分）'])

        l_max = get_l_max(age_group, score)
        problem = TCMInterventionProblem(s_0=s_0, l_max=l_max)
        res = minimize(problem, algorithm, ('n_gen', 100), seed=42, verbose=False)

        if res.F is not None:
            # 使用 TOPSIS 思想选择折中解
            F_norm = (res.F - res.F.min(axis=0)) / (res.F.max(axis=0) - res.F.min(axis=0) + 1e-8)
            best_idx = np.argmin(np.sqrt(np.sum(F_norm ** 2, axis=1)))
            best_x = res.X[best_idx]
            # 提取第一阶段(前两个月)平均干预强度作为回归靶向变量
            avg_early_intensity = (best_x[0] + best_x[1]) / 2.0
            y_target.append(avg_early_intensity)
        else:
            y_target.append(1.0)

    X_data['target_y'] = y_target
    X = X_data[features]
    y = X_data['target_y']

    # 基于 XGBoost 构建非线性映射引擎
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        objective='reg:squarederror'
    )
    xgb_model.fit(X, y)

    # 引入博弈论 SHAP 价值分解框架
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X)

    # 全局特征重要性 SHAP 蜂窝图
    plt.figure(figsize=(10, 6), dpi=300)
    shap.summary_plot(shap_values, X, show=False)
    plt.title("特征重要性 SHAP 蜂窝图", pad=20)
    plt.tight_layout()
    plt.savefig('SHAP_Summary_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 核心特征的 SHAP 单变量依赖图
    plt.figure(figsize=(8, 6), dpi=300)
    shap.dependence_plot("痰湿质", shap_values.values, X, show=False, interaction_index=None)
    plt.title("初始积分特征 SHAP 依赖图", pad=20)
    plt.tight_layout()
    plt.savefig('SHAP_Dependence_Plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"靶向映射与特征归因评估指标输出：")
    print(f"XGBoost 模型拟合优度): {xgb_model.score(X, y):.4f}")
    print(
        f"SHAP 基准值: {explainer.expected_value[0] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value:.4f}")