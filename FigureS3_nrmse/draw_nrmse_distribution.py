import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据文件
wbc_data = pd.read_csv('../data/all_wbc_nrmse.csv')
hgb_data = pd.read_csv('../data/all_hgb_nrmse.csv')
plt_data = pd.read_csv('../data/all_plt_nrmse.csv')
neut_data = pd.read_csv('../data/all_neut_nrmse.csv')

# 提取 nrmse 列
wbc_nrmse = wbc_data['nrmse']
hgb_nrmse = hgb_data['nrmse']
plt_nrmse = plt_data['nrmse']
neut_nrmse = neut_data['nrmse']

plt.rcParams['font.size'] = 12  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42
# 创建包含四个子图的图表
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# 绘制 WBC 的 NRMSE 分布图
sns.histplot(wbc_nrmse, color='skyblue', label='WBC', kde=True, bins=30, ax=axes[0, 0])
axes[0, 0].set_title('WBC', fontsize=16)
axes[0, 0].set_xlabel('NRMSE')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(False)

# 绘制 HGB 的 NRMSE 分布图
sns.histplot(hgb_nrmse, color='lightgreen', label='HGB', kde=True, bins=30, ax=axes[0, 1])
axes[0, 1].set_title('HGB', fontsize=16)
axes[0, 1].set_xlabel('NRMSE')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].grid(False)

# 绘制 PLT 的 NRMSE 分布图
sns.histplot(plt_nrmse, color='salmon', label='PLT', kde=True, bins=30, ax=axes[1, 0])
axes[1, 0].set_title('PLT', fontsize=16)
axes[1, 0].set_xlabel('NRMSE')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(False)

# 绘制 NEUT 的 NRMSE 分布图
sns.histplot(neut_nrmse, color='orchid', label='NEUT', kde=True, bins=30, ax=axes[1, 1])
axes[1, 1].set_title('NEUT', fontsize=16)
axes[1, 1].set_xlabel('NRMSE')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(False)

axes[0, 0].set_xlim(0, 2)
axes[0, 1].set_xlim(0, 2)
axes[1, 0].set_xlim(0, 2)
axes[1, 1].set_xlim(0, 2)

# 调整布局
plt.tight_layout(rect=[0, 0, 1, 0.96])

# 显示图表
plt.show()



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 读取数据文件
# data1 = pd.read_csv('C:/Users/xyy/Desktop/all_wbc_nrmse.csv')
#
# # 提取 nrmse 列
# nrmse_values = data1['nrmse']
#
# plt.rcParams['font.size'] = 12  # 设置colorbar字体大小
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["pdf.fonttype"] = 42
# # 绘制 NRMSE 分布图
# plt.figure(figsize=(10, 6))
# sns.histplot(nrmse_values, color='skyblue', label='WBC', kde=True, bins=30)  # binwidth=bin_width
#
# # 添加标题和标签
# plt.title('Distribution of NRMSE for WBC', fontsize=16)
# plt.xlabel('NRMSE', fontsize=14)
# plt.ylabel('Frequency', fontsize=14)
#
# # 优化布局
# plt.tight_layout()
#
# # 显示图表
# plt.show()

