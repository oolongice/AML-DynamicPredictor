### 这个要用虚拟环境deeplearning运行（其他文件不用改动首选虚拟环境）
### 先conda activate deeplearning     再python tabnet_best_params.py

import torch
import numpy as np
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import roc_auc_score
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # 确保图形在独立窗口显示
import seaborn as sns
import json

# 设置随机种子函数，用于控制所有随机操作的可复现性
def set_seed(seed=42):
    np.random.seed(seed)           # 控制numpy的随机操作（如数据采样）
    torch.manual_seed(seed)        # 控制PyTorch CPU运算的随机性（如模型初始化、训练）
    torch.cuda.manual_seed_all(seed)  # 控制PyTorch GPU运算的随机性
    torch.backends.cudnn.deterministic = True  # 确保CUDA卷积操作的确定性
    torch.backends.cudnn.benchmark = False     # 禁用CUDA的自动优化，确保结果可复现

# 在代码开始处设置全局随机种子，确保整个流程的可复现性
# 这将控制：
# 1. 数据预处理过程中的随机操作
# 2. 模型初始化时的随机参数
# 3. 训练过程中的随机操作（如dropout等）
set_seed()

# 加载预处理后的数据（这一步重要！）
X_train = np.load('../data/preprocessed_X_train_900.npy')
X_test = np.load('../data/preprocessed_X_test_900.npy')
X_external = np.load('../data/preprocessed_X_external_900.npy')
y_train = np.load('../data/preprocessed_y_train_900.npy')
y_test = np.load('../data/preprocessed_y_test_900.npy')
y_external = np.load('../data/preprocessed_y_external_900.npy')

# 从JSON文件读取特征顺序
with open('../data/feature_order_900.json', 'r') as f:
    feature_order = json.load(f)
    numerical_cols = feature_order['numerical_cols']
    categorical_columns = feature_order['categorical_columns']

# 重建特征顺序（数值特征在前，类别特征在后）
ordered_feature_cols = numerical_cols + categorical_columns

# 更新特征名称列表用于后续特征重要性分析
feature_names = ordered_feature_cols


# 加载最佳模型
model = TabNetClassifier(
)
model.load_model(f'../data/best_model_900_nd8_na32_g11_cd3_bs48_lr0006538.pt.zip')

# 在后续加载模型时，也加载特征重要性
feature_importances = np.load(f'../data/feature_importances_900_nd8_na32_g11_cd3_bs48_lr0006538.npy')

# 计算 AUC
y_pred_train = model.predict_proba(X_train)[:, 1]
y_pred_test = model.predict_proba(X_test)[:, 1]
y_pred_external = model.predict_proba(X_external)[:, 1]

auc_train = roc_auc_score(y_train, y_pred_train)
auc_test = roc_auc_score(y_test, y_pred_test)
auc_external = roc_auc_score(y_external, y_pred_external)

print(f"AUC (Train): {auc_train:.4f}")
print(f"AUC (Test): {auc_test:.4f}")
print(f"AUC (External): {auc_external:.4f}")

# 创建结果文件夹
result_dir = f"best_results_900"
os.makedirs(result_dir, exist_ok=True)

plt.rcParams['font.size'] = 15  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42

# 绘制混淆矩阵和ROC曲线
def plot_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{dataset_name}", fontweight='bold')
    plt.savefig(os.path.join(result_dir, f"confusion_matrix_{dataset_name}.eps"))
    plt.show()
    plt.close()


def plot_roc_curve(y_true, y_pred, dataset_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset_name}', fontweight='bold')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_dir, f"roc_curve_{dataset_name}.eps"))
    plt.show()
    plt.close()

# # # 因为要快速画特征贡献的图 暂时把这几个图注释掉了
# # 绘制训练集、测试集和外部验证集的混淆矩阵和ROC曲线
# plot_confusion_matrix(y_train, y_pred_train, "Training cohort")
# plot_roc_curve(y_train, y_pred_train, "Training cohort")
#
# plot_confusion_matrix(y_test, y_pred_test, "Internal validation cohort")
# plot_roc_curve(y_test, y_pred_test, "Internal validation cohort")
#
# plot_confusion_matrix(y_external, y_pred_external, "External validation cohort")
# plot_roc_curve(y_external, y_pred_external, "External validation cohort")


# 替换特征名称
feature_name_mapping = {
    'Pre_treatment_Mean_PLT': 'Mean value of PLT before treatment',
    'Pre_treatment_Mean_WBC': 'Mean value of WBC before treatment',
    'Pre_treatment_Mean_NEUT': 'Mean value of NEUT before treatment',
    'Pre_treatment_Mean_LYMPH': 'Mean value of LYMPH before treatment',
    'Pre_treatment_Mean_HGB': 'Mean value of HGB before treatment',
    'Admission_Value_PLT': 'PLT at admission',  ##'PLT value at admission',
    'Admission_Value_NEUT': 'NEUT value at admission',
    'Admission_Value_WBC': 'WBC value at admission',
    'Admission_Value_LYMPH': 'LYMPH value at admission',
    'Admission_Value_HGB': 'HGB value at admission',
    'B': 'CFB',
    # # 其他常见缩写的解释
    # 'UBIL': 'Unconjugated bilirubin',
    # 'DBIL': 'Direct bilirubin',
    # 'TBIL': 'Total bilirubin',
    # 'ALB': 'Albumin',
    # 'GLU': 'Glucose',
    # 'GGT': 'Gamma-glutamyl transferase',
    # 'CYSC': 'Cystatin C',
    # 'ProBNP': 'Pro-brain natriuretic peptide',
    # 'CRP': 'C-reactive protein',
    # 'SAA': 'Serum amyloid A',
    # 'PT': 'Prothrombin time',
    # 'APTT': 'Activated partial thromboplastin time',
    # 'LDL': 'Low-density lipoprotein',
    # 'PCT': 'Procalcitonin',
    # 'FIB-C': 'Fibrinogen-C'
}

# 精确匹配替换特征名
feature_names = [feature_name_mapping.get(name, name) for name in feature_names]

## 创建特征名称和重要性的映射
feature_importance_dict = dict(zip(feature_names, feature_importances))

# 按重要性排序
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 打印前20个特征的重要性
print("\n20 most important features:")
for feature_name, importance in sorted_features[:20]:
    print(f"{feature_name}: {importance:.4f}")

# 绘制前15个特征的重要性条形图  figsize=(15, 8)
top_n = 15
plt.figure(figsize=(12, 7))
features = [x[0] for x in sorted_features[:top_n]]
importances = [x[1] for x in sorted_features[:top_n]]

plt.bar(range(top_n), importances, color='skyblue')
plt.xticks(range(top_n), features, rotation=30, ha='right', fontsize=15, fontweight='bold')
plt.xlabel('Feature Name', fontsize=17, fontweight='bold')
plt.ylabel('Importance', fontsize=17, fontweight='bold')
# plt.title(f'Top {top_n} Important Features')
plt.tight_layout()  # 自动调整布局，防止标签被切割
# plt.savefig(os.path.join(result_dir, 'top_feature_importances.eps'))
plt.show()
# plt.close()