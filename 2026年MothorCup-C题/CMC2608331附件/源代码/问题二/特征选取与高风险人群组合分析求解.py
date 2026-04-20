import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.tree import DecisionTreeClassifier, plot_tree
from mlxtend.frequent_patterns import fpgrowth, association_rules

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
if __name__ == '__main__':

    # 路径
    risk_labels_path = r"C:\Users\jack\Desktop\三级风险分层结果.xlsx"
    raw_features_path = r"C:\Users\jack\Desktop\2026年第十六届MathorCup数学应用挑战赛赛题\C题\附件1：样例数据_异常值处理.xlsx"

    if not (os.path.exists(risk_labels_path) and os.path.exists(raw_features_path)):
        print("致命错误：未找到数据文件，请检查绝对路径。")
        exit()

    df_labels = pd.read_excel(risk_labels_path)
    df_raw = pd.read_excel(raw_features_path)

    # 特征空间重构与临床逻辑对齐，构建全量空间
    df_merged = pd.merge(df_labels[['样本ID', '风险等级', '风险类别编号']], df_raw, on='样本ID', how='inner')

    # 定义全量临床与中医特征池，作为白盒代理模型的自变量
    feature_cols = [
        '平和质', '气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质',
        '活动量表总分（ADL总分+IADL总分）', 'HDL-C（高密度脂蛋白）', 'LDL-C（低密度脂蛋白）',
        'TG（甘油三酯）', 'TC（总胆固醇）', '空腹血糖', '血尿酸', 'BMI'
    ]

    X_full = df_merged[feature_cols]
    Y_label = df_merged['风险类别编号']  # 0=Low, 1=Medium, 2=High

    # 基于 Gini 增益最大化与代价复杂性剪枝构建 CART 代理决策树
    cart_surrogate = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        min_samples_leaf=0.05,
        random_state=42
    )

    # 拟合全局代理模型
    cart_surrogate.fit(X_full, Y_label)

    # 打印全局代理树的分类逼近准确率
    print(f"CART 代理决策树对高阶拓扑分层标签的保真度: {cart_surrogate.score(X_full, Y_label):.4f}")

    # 全局三级风险特征阈值白盒转译决策树
    fig_tree, ax_tree = plt.subplots(figsize=(22, 12), dpi=300)
    plot_tree(
        cart_surrogate,
        feature_names=feature_cols,
        class_names=['低风险', '中风险', '高风险'],
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax_tree
    )
    plt.title('基于CART代理机制的全局三级风险临床特征显式判定树', fontsize=18)
    plt.tight_layout()
    plt.savefig('CART_Surrogate_Decision_Tree.png', dpi=300)
    plt.show()

    # 提取高风险子集，基于 FP-Growth 算法挖掘局部共病关联网络
    high_risk_subset = df_merged[(df_merged['风险类别编号'] == 2) & (df_merged['痰湿质'] >= 40)].copy()

    # 执行特征空间的临床分箱离散化转化为项集
    df_items = pd.DataFrame()
    df_items['高TG'] = high_risk_subset['TG（甘油三酯）'] >= 1.7
    df_items['高TC'] = high_risk_subset['TC（总胆固醇）'] >= 6.2
    df_items['低HDL'] = high_risk_subset['HDL-C（高密度脂蛋白）'] < 1.04
    df_items['高LDL'] = high_risk_subset['LDL-C（低密度脂蛋白）'] >= 3.1
    df_items['超重BMI'] = high_risk_subset['BMI'] >= 24
    df_items['痰湿严重'] = high_risk_subset['痰湿质'] >= 60
    df_items['低活动量'] = high_risk_subset['活动量表总分（ADL总分+IADL总分）'] < 40

    # 遍历 FP-Tree 提取满足最小支持度的频繁项集
    frequent_itemsets = fpgrowth(df_items, min_support=0.15, use_colnames=True)

    # 基于 Lift 提升度评估准则提取强关联致病规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

    # 按 Lift 降序提取核心的黄金共病组合规则
    top_rules = rules.sort_values(by='lift', ascending=False).head(15)

    print("\n高危人群核心共病特征黄金组合规则前五")
    print(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(5))

    # 高危人群核心共病特征关联网络图
    G = nx.DiGraph()

    # 提取规则前因后果构建有向网络边与权重
    for _, row in top_rules.iterrows():
        ant = ', '.join(list(row['antecedents']))
        con = ', '.join(list(row['consequents']))
        G.add_edge(ant, con, weight=row['lift'])

    fig_net, ax_net = plt.subplots(figsize=(12, 10), dpi=300)
    pos = nx.spring_layout(G, k=0.8, seed=42)

    # 计算节点出入度用于映射节点大小
    node_sizes = [3000 + 1000 * G.degree(node) for node in G.nodes()]

    # 绘制拓扑网络节点与边
    nx.draw_networkx_nodes(G, pos, node_color='#fdae61', node_size=node_sizes, alpha=0.9, edgecolors='k')
    edges = nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20, edge_color='#d7191c',
                                   width=[d['weight'] * 1.5 for (u, v, d) in G.edges(data=True)], alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=11, font_family='SimHei', font_weight='bold')

    plt.title('基于 FP-Growth 提升度度量的痰湿体质高风险人群共病特征关联拓扑网络', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('FP_Growth_Feature_Network.png', dpi=300)
    plt.show()