import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# # Load the data
# # data = pd.read_csv('D:/OneDrive/ModelParameterClassification/PatientData/nowdata/combined_patients_parameters.csv')
# # data = pd.read_csv('D:/OneDrive/ModelParameterClassification/PatientData/nowdata/all_data_granulocytopenia_period.csv')
# # data = pd.read_excel('C:/Users/xyy/Desktop/AML_allnew_data_newALT/AML_allnew_merged_patient_data_processed.xlsx')
# data = pd.read_excel('C:/Users/xyy/Desktop/nodata4ASTALT/AML_allnew_merged_patient_data_processed.xlsx')

# # Function to normalize a column
# def normalize(series):
#     return (series - series.min()) / (series.max() - series.min())
#
# # Normalize the parameters
# data['bN'] = normalize(data['B_NEUT'])
# data['bW'] = normalize(data['B_WBC'])
# data['bP'] = normalize(data['B_PLT'])
# data['bH'] = normalize(data['B_HGB'])
#
# # Calculate bsum
# data['bsum'] = data['bN'] + data['bW'] + data['bP'] + data['bH']
#
#
# # # 确定聚类数范围
# # n_components = np.arange(1, 11)
# # models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data[['bsum']])
# #           for n in n_components]
# #
# # # 计算BIC和AIC
# # bics = [model.bic(data[['bsum']]) for model in models]
# # aics = [model.aic(data[['bsum']]) for model in models]
# #
# # plt.rcParams['font.size'] = 15  # 设置colorbar字体大小
# # plt.rcParams['font.weight'] = 'bold'
# # plt.rcParams["axes.labelweight"] = "bold"
# # plt.rcParams["pdf.fonttype"] = 42
# # # 绘制BIC和AIC
# # plt.figure(figsize=(8, 4))
# # plt.plot(n_components, bics, label='BIC', linewidth = 3)
# # plt.plot(n_components, aics, label='AIC', linewidth = 3)
# # plt.legend(loc='best')
# # plt.xlabel('Number of components')
# # plt.ylabel('AIC Value')
# # plt.show()
# # #
# # # # 根据BIC和AIC选择最佳的n_components
# # print('best_bic:', n_components[np.argmin(bics)])
# # print('best_aic:', n_components[np.argmin(aics)])
# #
# # Fit a Gaussian Mixture Model with 4 components to bsum         best_bic = 4
# gmm = GaussianMixture(n_components=4, random_state=42)
# data['bsum_cluster'] = gmm.fit_predict(data['bsum'].values.reshape(-1, 1))
#
# # # # 计算每个样本的分类概率
# # # probs = gmm.predict_proba(data['bsum'].values.reshape(-1, 1))
# # #
# # # # 将概率添加到DataFrame
# # # data['prob_severe'] = probs[:, gmm.means_.flatten().argsort().argsort()[0]]
# # # data['prob_moderate'] = probs[:, gmm.means_.flatten().argsort().argsort()[1]]
# # #
# # # # 筛选模糊分类的病人，例如两个类别的概率都大于0.4
# # # ambiguous_patients = data[(data['prob_severe'] > 0.5) | (data['prob_moderate'] > 0.5)]
# # # # ambiguous_patients = data[abs(data['prob_severe'] - data['prob_moderate'])< 0.2]
# # # # 输出这些病人的相关信息
# # # print(ambiguous_patients[['id', 'prob_severe', 'prob_moderate']])
# #
# # Get the means of bsum for each cluster
# cluster_means = gmm.means_.flatten()
#
# # Create a mapping from old cluster labels to the suppression levels based on the order of means
# suppression_levels = ['Severe','Moderate','Mild','Minimal']  # 假设更高的均值对应于轻度
# ordered_labels = cluster_means.argsort().argsort()  # 排序并获取索引
# label_mapping = {i: suppression_levels[label] for i, label in enumerate(ordered_labels)}
#
# # Map the cluster labels to the suppression levels
# data['bsum_cluster_label'] = data['bsum_cluster'].map(label_mapping)
# # 统计每个类的样本数量
# cluster_counts = data['bsum_cluster_label'].value_counts()
# print(cluster_counts)
# #
# # # Create a new DataFrame with PatientID and classification results
# patient_classification = data[['id', 'bsum','bsum_cluster', 'bsum_cluster_label']]
# print(patient_classification.head())
# # # #Save this new DataFrame to a CSV file
# # patient_classification.to_csv('C:/Users/xyy/Desktop/4cluster_bsum_classification_results.csv', index=False)
#



# # 手动将Moderate，Mild，Minimal这3类合并为mild类(共207个病人)，这样减少一些类别不均衡。然后读取病人分成2类的结果画图
classification_results = pd.read_csv('../data/4to2cluster_bsum_classification_results.csv')

# 分2类  画图
plt.rcParams['font.size'] = 10  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42
# Plot the distribution of bsum with different colors for clusters


# #Determine the bin width by finding the range of 'bsum' and dividing by desired number of bins
# bin_range = classification_results['bsum'].max() - classification_results['bsum'].min()
# desired_bins = 50  # You can adjust this number as needed
# bin_width = bin_range / desired_bins

# Plot the distribution of bsum with different colors for clusters
plt.figure(figsize=(10, 6))
sns.histplot(classification_results[classification_results['new_bsum_cluster_label'] == 'Severe']['bsum'], color='lightcoral', label='Severe', kde=True, bins=30)  # binwidth=bin_width
sns.histplot(classification_results[classification_results['new_bsum_cluster_label'] == 'Mild']['bsum'], color='mediumseagreen', label='Mild', kde=True, bins=70)
# plt.title('Distribution of $B_{sum}$ with Suppression Levels', fontsize=15, fontweight='bold')
plt.xlabel('$B_{sum}$', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.legend(title='Suppression Level')
plt.tight_layout()
plt.show()



# # 画四级骨髓抑制持续时间在2个类之间的差异
from scipy.stats import shapiro, mannwhitneyu
data1 = pd.read_excel('../data/AML_allnew_merged_patient_data_processed.xlsx')
# # Plotting boxplots for HGB, WBC, NEUT, and PLT across different clusters
# # Duration of granulocytopenia
plt.rcParams['font.size'] = 12  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
variables = ['NEUT', 'PLT', 'HGB', 'WBC']
variable_labels = ['Duration of granulocytopenia', 'Duration of thrombocytopenic crisis', 'Duration of anemia', 'Duration of leukopenia']
suppression_order = ['Severe', 'Mild']  # 指定顺序

for ax, var, label in zip(axes.flatten(), variables, variable_labels):
    sns.boxplot(x='bsum_cluster_label', y=var, data=data1, ax=ax, palette='viridis', order=suppression_order, showfliers=False)
    # Perform Mann-Whitney U test
    severe_group = data1[data1['bsum_cluster_label'] == 'Severe'][var]
    mild_group = data1[data1['bsum_cluster_label'] == 'Mild'][var]
    u_stat, p_value = mannwhitneyu(severe_group, mild_group)

    # Annotate p-value inside the plot
    ax.text(0.5, 23, f"p-value = {p_value:.4e}", ha='center', va='bottom', color='k', fontsize=12,
            fontweight='bold')

    # ax.set_title(f'Boxplot of {var} across Clusters', fontsize=15, fontweight='bold')
    ax.set_xlabel(' ')  # Clearing out the default x-label
    ax.set_ylabel(label)  # Using custom y-labels

plt.tight_layout()
plt.show()






# # 确定聚类数范围
# n_components = np.arange(1, 11)
# models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(data[['bsum']])
#           for n in n_components]
#
# # 计算BIC和AIC
# # bics = [model.bic(data[['bsum']]) for model in models]
# aics = [model.aic(data[['bsum']]) for model in models]
#
# plt.rcParams['font.size'] = 15  # 设置colorbar字体大小
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["pdf.fonttype"] = 42
# # 绘制BIC和AIC
# plt.figure(figsize=(8, 4))
# # plt.plot(n_components, bics, label='BIC')
# plt.plot(n_components, aics, label='AIC', linewidth = 3)
# plt.legend(loc='best')
# plt.xlabel('Number of components')
# plt.ylabel('AIC Value')
# plt.show()
# #
# # # 根据BIC和AIC选择最佳的n_components
# # print('best_bic:', n_components[np.argmin(bics)])
# print('best_aic:', n_components[np.argmin(aics)])


# # 使用Gaussian Mixture Model进行聚类
# gmm = GaussianMixture(n_components=4, random_state=42)
# data['bsum_cluster'] = gmm.fit_predict(data['bsum'].values.reshape(-1, 1))
# probs = gmm.predict_proba(data['bsum'].values.reshape(-1, 1))
#
# # 将概率添加到DataFrame中
# prob_columns = [f'prob_cluster_{i}' for i in range(probs.shape[1])]
# for i, col in enumerate(prob_columns):
#     data[col] = probs[:, i]
#
# # 计算最高概率和第二高概率
# sorted_probs = np.sort(probs, axis=1)  # 按行排序概率
# max_probs = sorted_probs[:, -1]  # 最高概率
# second_max_probs = sorted_probs[:, -2]  # 第二高概率
# prob_differences = max_probs - second_max_probs  # 最高概率和第二高概率的差异
#
# # 同时满足两个条件的筛选：最大概率大于0.7且最高概率至少比第二高概率高出0.2
# threshold_max_prob = 0.7
# threshold_difference = 0.2
# filtered_indices = (max_probs > threshold_max_prob) & (prob_differences >= threshold_difference)
# filtered_samples = data[filtered_indices]
#
# print("满足条件的样本数量:", filtered_samples.shape[0])
#
# filtered_samples.to_csv('C:/Users/xyy/Desktop/gmm_filtered_data.csv', index=False)








# # 计算每个样本的两个最高概率之间的差异
# prob_diffs = probs.copy()
# prob_diffs.sort(axis=1)
# prob_diffs = prob_diffs[:, -1] - prob_diffs[:, -2]  # 计算最高概率和次高概率之间的差异
#
# # 将差异加入原始数据集中
# data['prob_diff'] = prob_diffs
#
# # 筛选出差异较小的样本
# # 这里的阈值可以根据实际情况调整，阈值越小，被认为是边缘的样本越接近两个类别的界限
# threshold = 0.3  # 示例阈值
# borderline_patients = data[data['prob_diff'] <= threshold]
#
# print("边缘病人数量:", borderline_patients.shape[0])
# print("边缘病人示例:")
# print(borderline_patients[['id', 'bsum', 'prob_diff', 'bsum_cluster_label']].head())

# 可选：将边缘病人信息保存到CSV文件
# borderline_patients.to_csv('C:/Users/xyy/Desktop/borderline_patients.csv', index=False)



# # 分4类 画图
# # Create a new DataFrame with PatientID and classification results
# patient_classification = data[['id', 'bsum_cluster', 'bsum_cluster_label']]
# print(patient_classification.head())
# # Save this new DataFrame to a CSV file
# patient_classification.to_csv('D:/OneDrive/ModelParameterClassification/PatientData/nowdata/alldata_bsum_classification_results.csv', index=False)
#
# # Plot the distribution of bsum with different colors for clusters
# plt.figure(figsize=(10, 6))
# colors = ['blue', 'orange', 'green','red']
# for cluster_label in suppression_levels:
#     cluster_data = data[data['bsum_cluster_label'] == cluster_label]['bsum']
#     sns.histplot(cluster_data, bins=20, kde=True, color=colors[ordered_labels[suppression_levels.index(cluster_label)]], label=cluster_label)
# plt.title('Distribution of bsum with Suppression Levels', fontsize=15, fontweight='bold')
# plt.xlabel('bsum')
# plt.ylabel('Frequency')
# plt.legend(title='Suppression Level')
# plt.tight_layout()
# plt.show()
#
# # Plotting boxplots for HGB, WBC, NEUT, and PLT across different clusters
# fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# variables = ['NEUT', 'PLT', 'HGB','WBC']
# suppression_order = ['Severe', 'Moderate', 'Mild', 'Minimal']  # 指定顺序
# for ax, var in zip(axes.flatten(), variables):
#     sns.boxplot(x='bsum_cluster_label', y=var, data=data, ax=ax, palette='viridis', order=suppression_order, showfliers=False)
#     ax.set_title(f'Boxplot of {var} across Clusters', fontsize=15, fontweight='bold')
#     ax.set_xlabel(' ')
#     ax.set_ylabel(var)
# plt.tight_layout()
# plt.show()





# # 用kmeans聚类
# # 使用肘部方法寻找最佳的k值    现在选取的是4
# inertia = []
# k_range = range(1, 8)
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(data[['bsum']])
#     inertia.append(kmeans.inertia_)
#
# # 绘制肘部曲线
# plt.figure(figsize=(8, 4))
# plt.plot(k_range, inertia, marker='o')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.title('Elbow Method For Optimal k')
# plt.show()


# # Choose 4 as the number of clusters
# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(data[['bsum']])
#
# # Assign the cluster labels to the data
# data['Cluster'] = kmeans.labels_
#
# # Plot the distribution of the 'bsum' feature for each cluster
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=data, x='bsum', y='bsum', hue='Cluster', palette='viridis', legend='full')
# plt.title('Distribution of Patients Across Clusters Based on bsum')
# plt.xlabel('bsum (Normalized)')
# plt.ylabel('bsum (Normalized)')
# plt.show()
#
# # Save the classified results to a new CSV file
# data.to_csv('D:/OneDrive/ModelParameterClassification/PatientData/nowdata/bsum_classification_results.csv', index=False)









