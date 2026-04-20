import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TCMInterventionProblem(ElementwiseProblem):
    def __init__(self, s_0, l_max):
        self.s_0 = s_0
        self.l_max = l_max
        # 决策变量：x1~x6为干预强度，y1~y6为每周频率
        xl = [1] * 6 + [1] * 6
        xu = [self.l_max] * 6 + [10] * 6
        super().__init__(n_var=12, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        s_t = self.s_0
        total_cost = 0.0

        for t in range(6):
            x_t = int(round(x[t]))
            y_t = int(round(x[t + 6]))

            # C_{TCM}(S_t) = 30 (S_t<=58), 80 (59<=S_t<=61), 130 (S_t>=62)
            if s_t <= 58:
                c_tcm = 30
            elif s_t <= 61:
                c_tcm = 80
            else:
                c_tcm = 130

            # 建立单次活动成本映射
            c_unit_map = {1: 3, 2: 5, 3: 8}
            c_unit = c_unit_map[x_t]

            # C_{month, t} = C_{TCM}(S_t) + 4 * y_t * C_{unit}(x_t)
            c_month = c_tcm + 4 * y_t * c_unit
            total_cost += c_month

            # r_t = 0 (y_t<5); x_t*3% + (y_t-5)*1% (y_t>=5)
            if y_t < 5:
                r_t = 0.0
            else:
                r_t = x_t * 0.03 + (y_t - 5) * 0.01

            # S_{t+1} = S_t * (1 - r_t)
            s_t = s_t * (1 - r_t)

            # 0 <= S_t <= 100
            s_t = max(0.0, min(100.0, s_t))

        # f_1 = S_7, f_2 = C_{total}
        out["F"] = [s_t, total_cost]
        # g_1 = \sum C_{month, t} - 2000 <= 0
        out["G"] = [total_cost - 2000.0]

def get_l_max(age_group, score):
    # L_{age} 约束逻辑
    if age_group in [1, 2]:
        l_age = 3
    elif age_group in [3, 4]:
        l_age = 2
    else:
        l_age = 1

    # L_{score} 约束逻辑
    if score < 40:
        l_score = 1
    elif score < 60:
        l_score = 2
    else:
        l_score = 3

    # L_{max}^{(t)} = \min(L_{age}, L_{score})
    return min(l_age, l_score)


def plot_pareto_front(res, patient_id):
    F = res.F
    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(F[:, 1], F[:, 0], color='blue', alpha=0.7, edgecolors='k')
    plt.title(f'患者 ID: {patient_id} 帕累托前沿散点图')
    plt.xlabel('6个月总干预成本')
    plt.ylabel('期末痰湿体质积分')
    plt.axvline(x=2000, color='r', linestyle='--', label='2000元预算红线')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Patient_{patient_id}_Pareto.png', dpi=300)
    plt.close()


def plot_trajectory(best_x, s_0, patient_id):
    s_history = [s_0]
    tcm_costs = []
    act_costs = []

    s_t = s_0
    for t in range(6):
        x_t = int(round(best_x[t]))
        y_t = int(round(best_x[t + 6]))

        # C_{TCM}(S_t) 计算模型重构
        if s_t <= 58:
            c_tcm = 30
        elif s_t <= 61:
            c_tcm = 80
        else:
            c_tcm = 130
        tcm_costs.append(c_tcm)

        c_unit_map = {1: 3, 2: 5, 3: 8}
        c_unit = c_unit_map[x_t]
        c_act = 4 * y_t * c_unit
        act_costs.append(c_act)

        # S_{t+1} = S_t * (1 - r_t)
        r_t = 0.0 if y_t < 5 else x_t * 0.03 + (y_t - 5) * 0.01
        s_t = s_t * (1 - r_t)
        s_history.append(s_t)

    months = np.arange(1, 7)
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    ax1.bar(months, tcm_costs, color='#FFA07A', label='中医调理成本', alpha=0.8)
    ax1.bar(months, act_costs, bottom=tcm_costs, color='#87CEFA', label='活动训练成本', alpha=0.8)
    ax1.set_xlabel('干预月份')
    ax1.set_ylabel('单月开销', color='k')
    ax1.set_ylim(0, 450)

    ax2 = ax1.twinx()
    ax2.plot(range(0, 7), s_history, color='red', marker='o', linewidth=2.5, label='痰湿积分状态演化')
    ax2.set_ylabel('痰湿体质积分', color='red')
    ax2.axhline(y=62, color='gray', linestyle='-.', alpha=0.5, label='130元档阈值')
    ax2.axhline(y=59, color='gray', linestyle=':', alpha=0.5, label='80元档阈值')

    fig.suptitle(f'患者 ID: {patient_id} 动态干预轨迹与成本消耗双轴图')
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    plt.tight_layout(rect=[0, 0.03, 1, 0.9])
    plt.savefig(f'Patient_{patient_id}_Trajectory.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    file_path = r'C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx'

    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        print("未找到Excel文件，请检查绝对路径是否正确。")
        exit()

    # 确诊痰湿体质，体质分型标签=5
    target_ids = [1, 2, 3]

    # 构建进化算法算子体系
    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=0.9, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=0.1, vtype=float, repair=RoundingRepair())
    )

    for pid in target_ids:
        patient_data = df[df['样本ID'] == pid]
        if patient_data.empty:
            continue

        row = patient_data.iloc[0]
        s_0 = float(row.get('痰湿积分', row.get('中医体质分型_积分', 70.0)))
        age_group = int(row.get('年龄组', 2))
        score = float(row.get('活动量表总分', 65.0))

        l_max = get_l_max(age_group, score)

        problem = TCMInterventionProblem(s_0=s_0, l_max=l_max)

        res = minimize(problem, algorithm, ('n_gen', 200), seed=42, verbose=False)

        if res.F is not None:
            plot_pareto_front(res, pid)

            # 使用理想点法在 Pareto 解集中选出折中解
            F_norm = (res.F - res.F.min(axis=0)) / (res.F.max(axis=0) - res.F.min(axis=0) + 1e-8)
            distances = np.sqrt(np.sum(F_norm ** 2, axis=1))
            best_idx = np.argmin(distances)

            best_x = res.X[best_idx]
            best_f = res.F[best_idx]

            plot_trajectory(best_x, s_0, pid)

            print(f"患者 ID: {pid} 最优求解结果:")
            print(f"初始积分: {s_0:.1f}, 生理耐受度最高强度: {l_max}级")
            print(f"推荐强度序列: {[int(x) for x in best_x[:6]]}")
            print(f"推荐频率序列: {[int(x) for x in best_x[6:]]}")
            print(f"期末痰湿积分: {best_f[0]:.2f}")
            print(f"六个月总成本: {best_f[1]:.2f} 元\n")