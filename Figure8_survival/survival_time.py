import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, mannwhitneyu

# Load the dataset
file_path = '../data/data3_survival_data.xlsx'
data = pd.read_excel(file_path)

# Separate the data into two groups
mild_group = data[data['cluster_label'] == 'Mild']
severe_group = data[data['cluster_label'] == 'Severe']

# Perform Shapiro-Wilk test for normality
shapiro_mild = shapiro(mild_group['survival_time'])
shapiro_severe = shapiro(severe_group['survival_time'])

print("Shapiro-Wilk Test Results:")
print(f"Mild group: W-Statistic = {shapiro_mild[0]:.4f}, p-value = {shapiro_mild[1]:.4e}")
print(f"Severe group: W-Statistic = {shapiro_severe[0]:.4f}, p-value = {shapiro_severe[1]:.4e}")

# Perform Mann-Whitney U test
u_stat, p_value_mannwhitney = mannwhitneyu(mild_group['survival_time'], severe_group['survival_time'])


plt.rcParams['font.size'] = 15  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42
# Plot box plot with Mann-Whitney U test p-value annotation
plt.figure(figsize=(10, 6))
sns.boxplot(x='cluster_label', y='survival_time', data=data)
# Annotate the p-value inside the plot
x1, x2 = 0, 1
y, h, col = data['survival_time'].max() + 10, 50, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1 + x2) * .5, y + h, f"p-value = {p_value_mannwhitney:.4e}", ha='center', va='bottom', color=col)

plt.xlabel('Cluster')
plt.ylabel('In-Hospital Survival Time (days)')
plt.show()

print(f"Mann-Whitney U Test p-value: {p_value_mannwhitney:.4e}")
