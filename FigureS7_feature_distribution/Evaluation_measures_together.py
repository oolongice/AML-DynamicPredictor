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
    """Compute GOF and create a single multi-panel figure for all features.

    - Continuous features: KDEs of Virtual vs Original (filled, transparent)
    - Discrete features: side-by-side bars of category proportions
    - Shared legend at figure level
    - Saves one PDF in the results directory
    """
    # Collect features that actually exist in the original data
    features = []
    for f in continuous_columns:
        if f in original_data.columns:
            features.append((f, 'cont'))
    for f in discrete_columns:
        if f in original_data.columns:
            features.append((f, 'disc'))

    # Initialize GOF results
    gof_results = []

    # Grid size: ~4 plots per row
    N = len(features)
    if N == 0:
        # Nothing to plot, still save empty results
        gof_results_df = pd.DataFrame(columns=['Feature', 'KS Statistic', 'P-value'])
        gof_results_df.to_excel(results_path, index=False)
        return gof_results_df

    ncols = 4
    nrows = int(np.ceil(N / ncols))

    # Figure setup
    figsize = (20, 3 * nrows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)

    # Consistent colors and labels
    sim_color = 'tab:orange'
    orig_color = 'tab:blue'
    sim_label = 'Virtual Data'
    orig_label = 'Original Data'

    # Local import for legend patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=sim_color, edgecolor='none', label=sim_label, alpha=0.5),
                      Patch(facecolor=orig_color, edgecolor='none', label=orig_label, alpha=0.5)]

    # Helper to map feature display name
    def display_name_of(col):
        return feature_name_mapping.get(col, col)

    # Draw each subplot
    for idx, (feature, ftype) in enumerate(features):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # Prepare data
        s_sim = simulated_data[feature].dropna()
        s_orig = original_data[feature].dropna()

        # KS test (keep as-is for compatibility)
        try:
            gof_statistic, gof_p_value = stats.ks_2samp(s_sim, s_orig)
        except Exception:
            # If data types are incompatible for KS, set NaNs
            gof_statistic, gof_p_value = np.nan, np.nan
        gof_results.append((feature, gof_statistic, gof_p_value))

        title_name = display_name_of(feature)

        if ftype == 'cont':
            # Continuous: KDEs with fallback to hist if KDE not feasible
            plotted = False
            try:
                if s_sim.var() > 0:
                    sns.kdeplot(s_sim, ax=ax, fill=True, alpha=0.4, color=sim_color, linewidth=1.2)
                    plotted = True
                else:
                    ax.hist(s_sim, bins=10, density=True, alpha=0.4, color=sim_color)
                if s_orig.var() > 0:
                    sns.kdeplot(s_orig, ax=ax, fill=True, alpha=0.4, color=orig_color, linewidth=1.2)
                    plotted = True
                else:
                    ax.hist(s_orig, bins=10, density=True, alpha=0.4, color=orig_color)
            except Exception:
                # Fallback: histograms if KDE fails
                ax.hist(s_sim, bins=10, density=True, alpha=0.4, color=sim_color)
                ax.hist(s_orig, bins=10, density=True, alpha=0.4, color=orig_color)

            ax.set_xlabel(title_name, fontweight='bold')
            ax.set_ylabel('Density', fontweight='bold')
        else:
            # Discrete: side-by-side bars of proportions
            sim_counts = s_sim.value_counts(normalize=True)
            orig_counts = s_orig.value_counts(normalize=True)
            # Category order: appearance in original then simulated
            categories = list(dict.fromkeys(list(s_orig.dropna().unique()) + list(s_sim.dropna().unique())))
            sim_vals = np.array([sim_counts.get(cat, 0.0) for cat in categories], dtype=float)
            orig_vals = np.array([orig_counts.get(cat, 0.0) for cat in categories], dtype=float)

            x = np.arange(len(categories))
            width = 0.4
            ax.bar(x - width/2, sim_vals, width, color=sim_color)
            ax.bar(x + width/2, orig_vals, width, color=orig_color)
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in categories], rotation=45, ha='right')
            ax.set_ylim(0, max(1.0, float(np.nanmax([sim_vals.max() if sim_vals.size else 0, orig_vals.max() if orig_vals.size else 0])) * 1.15))
            ax.set_xlabel(title_name, fontweight='bold')
            ax.set_ylabel('Proportion', fontweight='bold')

        # No per-axes legends

    # Turn off any unused axes
    total_axes = nrows * ncols
    for j in range(N, total_axes):
        r, c = divmod(j, ncols)
        axes[r][c].axis('off')

    # Shared legend at figure level (bottom)
    fig.subplots_adjust(bottom=0.08, wspace=0.3, hspace=0.4)
    fig.legend(handles=legend_handles, labels=[sim_label, orig_label],
               loc='lower center', ncol=2, frameon=False)

    # Save combined figure in results directory (PDF only)
    out_dir = os.path.dirname(results_path) if results_path else '.'
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, 'all_features_comparison.pdf')
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Prepare and save GOF results
    gof_results_df = pd.DataFrame(gof_results, columns=['Feature', 'KS Statistic', 'P-value'])
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
base_images_dir = 'Result/Virtual_and_Original_DistributionPlots(png)'
os.makedirs(base_images_dir, exist_ok=True)
results_dir = 'Result/Figure_comparison'

original_data_path = '../data/nodata2_2cluster_training_set.xlsx'
simulated_data_path = '../data/nodata2_2cluster_simulated_training_patient_set_700.xlsx'

main(simulated_data_path, original_data_path, results_dir, base_images_dir)

# %%
