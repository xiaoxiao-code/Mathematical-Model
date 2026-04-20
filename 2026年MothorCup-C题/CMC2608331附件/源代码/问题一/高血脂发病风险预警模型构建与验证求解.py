import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import plot_optimization_history, plot_contour

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == '__main__':
    # 读取数据
    data_path = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"
    df = pd.read_excel(data_path)

    # 设定二分类标签与17项关键特征
    target_col = '高血脂症二分类标签'
    selected_features = [
        'ADL吃饭', 'ADL用厕', 'IADL理财', 'TC（总胆固醇）', 'IADL交通',
        'ADL洗澡', 'BMI', 'IADL购物', '空腹血糖', 'HDL-C（高密度脂蛋白）',
        'ADL步行', 'IADL服药', '血尿酸', 'LDL-C（低密度脂蛋白）',
        'TG（甘油三酯）', 'ADL穿衣', 'IADL做饭'
    ]

    # 构建特征矩阵 X 与标签向量 y
    X = df[selected_features]
    y = df[target_col]

    # 按 8:2 划分训练集与测试集保障泛化性能评估
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 动态计算代价敏感交叉熵惩罚权重
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 构建贝叶斯寻优目标函数
    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'scale_pos_weight': pos_weight,
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'verbose': -1,
            'random_state': 42
        }

        # 构建 LightGBM 基分类器
        model = lgb.LGBMClassifier(**params)

        # 使用 5 折分层交叉验证评估 AUC 增量
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

        return scores.mean()

    # 实例化 Optuna 寻优研究对象寻求 AUC 最大化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # 提取最优参数组合训练最终诊断模型
    best_params = study.best_params
    best_params['objective'] = 'binary'
    best_params['scale_pos_weight'] = pos_weight
    best_params['random_state'] = 42
    best_params['verbose'] = -1

    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)

    # 计算测试集的预测概率与 ROC 曲线坐标
    y_pred_prob = final_model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # 提取全局预警增益 Gain 特征重要度
    feature_importances = final_model.booster_.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'特征': X.columns, 'Gain重要度': feature_importances})
    imp_df = imp_df.sort_values(by='Gain重要度', ascending=True)

    # 基于 Optuna 贝叶斯优化的超参数寻优历史与等高线图
    fig1 = plt.figure(figsize=(16, 6), dpi=300)

    ax1 = fig1.add_subplot(121)
    plot_optimization_history(study, target_name="ROC-AUC 得分").set_title("贝叶斯优化收敛历史图", fontsize=14)

    ax2 = fig1.add_subplot(122)
    # 绘制核心超参数的高斯过程期望增量等高线
    plot_contour(study, params=['num_leaves', 'learning_rate']).set_title("参数空间等高线投影图", fontsize=14)

    plt.tight_layout()
    plt.savefig("01_Bayesian_Optimization_Analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 代价敏感 LightGBM 模型的 ROC 曲线与重要度条形图
    fig2 = plt.figure(figsize=(16, 7), dpi=300)

    # 绘制 ROC 曲线评估分类效能
    ax3 = fig2.add_subplot(121)
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'代价敏感模型 (AUC = {roc_auc:.4f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('假阳性率', fontsize=12)
    ax3.set_ylabel('真阳性率', fontsize=12)
    ax3.set_title('高血脂分类预警 ROC 评估曲线', fontsize=15, pad=15)
    ax3.legend(loc="lower right", fontsize=12)

    # 绘制特征增益 Gain 预警重要度条形图
    ax4 = fig2.add_subplot(122)
    ax4.barh(imp_df['特征'], imp_df['Gain重要度'], color='mediumseagreen')
    ax4.set_xlabel('全局预警重要度', fontsize=12)
    ax4.set_title('核心特征高血脂预警增益条形图', fontsize=15, pad=15)

    plt.tight_layout()
    plt.savefig("02_ROC_and_Feature_Gain.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 打印结果
    print(f"非平衡数据权重: {pos_weight:.2f}")
    print(f"贝叶斯寻优最佳参数组合: {study.best_params}")
    print(f"测试集 ROC-AUC 验证得分: {roc_auc:.4f}")
    print("\n依据 Gain 增益的高血脂预警核心指标全部排名：")
    all_features = imp_df.iloc[::-1]
    for rank, (idx, row) in enumerate(all_features.iterrows(), 1):
        print(f" 第 {rank} 名: {row['特征']} (信息增益: {row['Gain重要度']:.2f})")