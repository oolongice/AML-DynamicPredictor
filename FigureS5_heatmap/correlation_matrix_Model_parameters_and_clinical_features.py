import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
data = pd.read_excel('../data/AML_allnew_merged_patient_data_processed.xlsx')

# List of features to include in the correlation matrix
features_to_include = [
    'Pre_treatment_Mean_PLT', 'Pre_treatment_Mean_HGB', 'Pre_treatment_Mean_WBC', 'Pre_treatment_Mean_NEUT',
    'Admission_Value_PLT', 'Admission_Value_HGB', 'Admission_Value_WBC', 'Admission_Value_NEUT',
    'SA','LDH','pct','INR','SAA','GLU','CRP','GLB','A/G',
    'ALB', 'CHE', 'B', 'TP','β2-MG'
    # 'CHE', 'B', 'LDH1', 'ALB', 'CRP', 'SOD',
    # 'FIB-C', 'PA', 'DBIL', 'TP', 'C1q', 'CYSC',
    # 'BUN', 'SAA', 'SA', 'UBIL','A/G'
]

# Dictionary for renaming features
feature_names = {
    'Pre_treatment_Mean_PLT': 'Mean PLT before treatment',
    'Pre_treatment_Mean_HGB': 'Mean HGB before treatment',
    'Pre_treatment_Mean_WBC': 'Mean WBC before treatment',
    'Pre_treatment_Mean_NEUT': 'Mean NEUT before treatment',
    'Admission_Value_PLT': 'PLT at admission',
    'Admission_Value_HGB': 'HGB at admission',
    'Admission_Value_WBC': 'WBC at admission',
    'Admission_Value_NEUT': 'NEUT at admission',
    'B': 'CFB',
    # 'SA': 'SA','LDH': 'LDH','pct': 'pct','INR': 'INR','SAA': 'SAA','GLU': 'GLU','CRP': 'CRP','GLB': 'GLB','A/G':'A/G',
    # 'ALB': 'ALB', 'CHE': 'CHE',  'TP': 'TP','β2-MG'
}

# Calculate the correlation matrix for the selected features
correlation_matrix = data[features_to_include].corr()

# Rename the rows and columns of the correlation matrix
correlation_matrix.rename(columns=feature_names, index=feature_names, inplace=True)

# plt.rcParams['font.size'] = 12  # 设置colorbar字体大小
# plt.rcParams['font.weight'] = 'bold'
# plt.rcParams["axes.labelweight"] = "bold"
# plt.rcParams["pdf.fonttype"] = 42
# Plot the correlation matrix
plt.figure(figsize=(16, 16))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
# plt.title('Correlation Matrix of Specified Features')
plt.xticks(rotation=30, ha='right', fontweight='bold', fontsize=12)
plt.yticks(rotation=0, fontweight='bold', fontsize=12)
plt.tight_layout()  # Adjust the layout to fit the figure size
plt.show()

# # Save the figure
# plt.savefig('D:/OneDrive/ModelParameterClassification/Results/Model_parameters_and_clinical_features_correlation_matrix.png')
# plt.close()  # Close the figure after saving to avoid displaying it in the notebook
