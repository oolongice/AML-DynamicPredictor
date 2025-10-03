#%%
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import os

# # 定义处理异常值的函数
# def handle_outliers(data, continuous_columns):
#     for column in continuous_columns:
#         Q1 = data[column].quantile(0.25)
#         Q3 = data[column].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#
#         data[column] = np.clip(data[column], lower_bound, upper_bound)
plt.rcParams['font.size'] = 15  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42

# 添加特征名称映射字典
feature_name_mapping = {
    'Pre_treatment_Mean_PLT': 'Mean PLT before treatment',
    'Pre_treatment_Mean_HGB': 'Mean HGB before treatment',
    'Pre_treatment_Mean_WBC': 'Mean WBC before treatment',
    'Pre_treatment_Mean_NEUT': 'Mean NEUT before treatment',
    'Admission_Value_PLT': 'PLT at admission',
    'Admission_Value_HGB': 'HGB at admission',
    'Admission_Value_WBC': 'WBC at admission',
    'Admission_Value_NEUT': 'NEUT at admission',
    'B': 'CFB'
}

# 连续变量用核密度估计 拟合分布 然后对比
def plot_and_save_density(sim_data, orig_data, feature, save_dir):
    if sim_data[feature].var() > 0 and orig_data[feature].var() > 0:
        plt.figure(figsize=(8, 6))
        sns.kdeplot(sim_data[feature], label='Virtual Data', fill=True)
        sns.kdeplot(orig_data[feature], label='Original Data', fill=True)

        # 使用映射后的特征名称（如果存在）
        display_name = feature_name_mapping.get(feature, feature)
        plt.title(f'Density distribution of {display_name}', fontweight='bold', fontsize=15)
        plt.xlabel(display_name, fontweight='bold', fontsize=15)
        plt.ylabel('Density', fontweight='bold', fontsize=15)
        plt.legend()

        # 保存时仍使用原始特征名
        safe_feature_name = feature.replace('/', '_or_')
        save_path = os.path.join(save_dir, f"{safe_feature_name}_density.png")
        plt.savefig(save_path)
        plt.close()

# 对离散特征绘制堆积条形图的函数
def plot_and_save_stacked_bar(sim_data, orig_data, feature, save_dir):
    # 获取特征在两个数据集中的所有类别
    all_categories = set(sim_data[feature].dropna().unique()).union(set(orig_data[feature].dropna().unique()))
    combined_counts_sim = {category: sim_data[feature].value_counts().get(category, 0)/len(sim_data) for category in all_categories}
    combined_counts_orig = {category: orig_data[feature].value_counts().get(category, 0)/len(orig_data) for category in all_categories}

    fig, ax = plt.subplots()
    # Stacking for original data
    bottom_original = 0
    for i, category in enumerate(all_categories):
        height = combined_counts_orig[category]  # Use counts for original data
        color = plt.cm.viridis(i / len(all_categories))  # Color based on index
        ax.bar('Original Data', height, bottom=bottom_original, color=color,
               label=f'{category}' if i == 0 else "")
        bottom_original += height

    # Stacking for simulate data
    bottom_new = 0
    for i, category in enumerate(all_categories):
        height = combined_counts_sim[category]
        color = plt.cm.viridis(i / len(all_categories))
        ax.bar('Virtual Data', height, bottom=bottom_new, color=color,
               label=f'{category}' if i == 0 else "")
        bottom_new += height

    # 使用映射后的特征名称（如果存在）
    display_name = feature_name_mapping.get(feature, feature)
    ax.set_ylabel('Percentage', fontweight='bold', fontsize=15)
    ax.set_title(f'Stacked bar chart of {display_name}', fontweight='bold', fontsize=15)
    ax.legend([f'{cat}' for cat in all_categories])

    # 修改文件名中的非法字符
    safe_feature_name = feature.replace('/', '_or_')
    # 使用os.path.join确保正确处理路径
    save_path = os.path.join(save_dir, f"{safe_feature_name}_density.png")
    plt.savefig(save_path)
    plt.close()

# 定义计算拟合优度并保存结果的函数
def compute_gof_and_plot(simulated_data, original_data, continuous_columns, discrete_columns, images_dir, results_path):
    # 初始化结果列表
    gof_results = []

    # 对每个连续特征进行Kolmogorov-Smirnov拟合优度检验，并绘制密度分布图
    for feature in continuous_columns:
        if feature in original_data.columns:
            gof_statistic, gof_p_value = stats.ks_2samp(simulated_data[feature], original_data[feature])
            gof_results.append((feature, gof_statistic, gof_p_value))

            # 绘制并保存密度分布图
            plot_and_save_density(simulated_data, original_data, feature, images_dir)

    # 对每个离散特征绘制并保存堆积条形图
    for feature in discrete_columns:
        # Assuming gof_results is defined earlier
        gof_statistic, gof_p_value = stats.ks_2samp(simulated_data[feature], original_data[feature])
        gof_results.append((feature, gof_statistic, gof_p_value))
        plot_and_save_stacked_bar(simulated_data, original_data, feature, images_dir)

    # 结果转换成DataFrame
    gof_results_df = pd.DataFrame(gof_results, columns=['Feature', 'KS Statistic', 'P-value'])

    # 保存完整的结果到CSV文件中
    gof_results_df.to_excel(results_path, index=False)
    return gof_results_df


# 主函数，用于加载数据和调用其他函数
def main(simulated_data_path, original_data_path, results_dir, base_images_dir):

    results_path = os.path.join(results_dir, f'gof_results.xlsx')

    simulated_data = pd.read_excel(simulated_data_path)
    original_data = pd.read_excel(original_data_path)
    # 定义要删除的列名列表
    columns_to_drop = [
        'bsum_cluster', 'HGB', 'WBC', 'NEUT', 'PLT', 'B_WBC', 'gamma_WBC', 'ktr_WBC', 'slopeA_WBC', 'slopeD_WBC',
        'B_HGB', 'gamma_HGB', 'ktr_HGB', 'slopeA_HGB', 'slopeD_HGB', 'B_NEUT', 'gamma_NEUT', 'ktr_NEUT', 'slopeA_NEUT',
        'slopeD_NEUT', 'B_PLT', 'gamma_PLT', 'ktr_PLT', 'slopeA_PLT', 'slopeD_PLT'
    ]

    # 删除指定的列
    original_data.drop(columns=columns_to_drop, inplace=True)

    # # 目前还是对整体数据在处理和比较
    specified_columns = [col for col in simulated_data.columns if
                         col not in ['id', 'bsum_cluster_label']]

    discrete_columns = ['DD','AST/ALT','ALP_categorized','ALT_categorized','AST_categorized','sex','HBsAg','HBeAg',
                        'Anti-HCV', 'HIV-Ab', 'Syphilis','BG', 'ABOZDX', 'ABOFDX', 'Rh', 'BGZGTSC'
    ]

    continuous_columns = [col for col in specified_columns if col not in discrete_columns]

    # # 假设 specified_columns, discrete_columns, continuous_columns 已经定义
    # # 处理异常值
    # handle_outliers(simulated_data, continuous_columns)

    # 计算拟合优度并绘图
    gof_results_df = compute_gof_and_plot(simulated_data, original_data, continuous_columns, discrete_columns, base_images_dir, results_path)

    # 显示部分结果
    print(gof_results_df.head())

# 创建保存图像的目录（如果尚不存在）
base_images_dir = '../Figure/Comparison/Virtual_and_Original_DistributionPlots(png)'
os.makedirs(base_images_dir, exist_ok=True)
results_dir = '../Figure/Comparison'

original_data_path = '../data/nodata2_2cluster_training_set.xlsx'
simulated_data_path = '../data/nodata2_2cluster_simulated_training_patient_set_700.xlsx'

main(simulated_data_path, original_data_path, results_dir, base_images_dir)
