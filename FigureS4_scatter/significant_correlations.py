import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
# data = pd.read_csv('D:/OneDrive/ModelParameterClassification/PatientData/nowdata/combined_patients_parameters.csv')
# data = pd.read_csv('C:/Users/xyy/Desktop/all_data_granulocytopenia_period.csv')
data = pd.read_excel('../data/AML_allnew_merged_patient_data_processed.xlsx')
# Initialize a dictionary to store the results
correlation_results = {
    'Feature': [],
    'Cell Type': [],
    'Correlation Coefficient': [],
    'P-value': []
}

# Dictionary mapping cell types to their respective features
features_for_cell_types = {
    'HGB': ['B_HGB', 'gamma_HGB', 'ktr_HGB', 'slopeA_HGB', 'slopeD_HGB'],
    'PLT': ['B_PLT', 'gamma_PLT', 'ktr_PLT', 'slopeA_PLT', 'slopeD_PLT'],
    'WBC': ['B_WBC', 'gamma_WBC', 'ktr_WBC', 'slopeA_WBC', 'slopeD_WBC'],
    'NEUT': ['B_NEUT', 'gamma_NEUT', 'ktr_NEUT', 'slopeA_NEUT', 'slopeD_NEUT']
}

# Loop through each cell type and its specified features to calculate the correlation coefficient and p-value
for cell_type, features in features_for_cell_types.items():
    for feature in features:
        # Check if both cell_type and feature columns are in the data
        if cell_type in data.columns and feature in data.columns:
            corr_coeff, p_value = scipy.stats.pearsonr(data[cell_type], data[feature])
            correlation_results['Feature'].append(feature)
            correlation_results['Cell Type'].append(cell_type)
            correlation_results['Correlation Coefficient'].append(corr_coeff)
            correlation_results['P-value'].append(p_value)

# Convert the results to a DataFrame
correlation_df = pd.DataFrame(correlation_results)
# print(correlation_df)
# Filter for statistically significant correlations (p < 0.05)
significant_correlations = correlation_df[correlation_df['P-value'] < 0.05].reset_index(drop=True)

# Display the significant correlations
print(significant_correlations)
# significant_correlations.to_csv('D:/OneDrive/ModelParameterClassification/Newresults(2cluster)/significant_correlations.csv')

# Initialize the plot with a certain size
plt.figure(figsize=(10, 10))
plt.rcParams['font.size'] = 12  # 设置colorbar字体大小
plt.rcParams['font.weight'] = 'bold'
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["pdf.fonttype"] = 42

# Set the number of subplots based on the number of significant correlations
num_plots = len(significant_correlations)
num_rows = 3  # You can adjust this as needed
num_cols = 3  # You can adjust this as needed

# Define a mapping for feature names to LaTeX formatted strings
latex_feature_names = {
    'B_HGB': '$B_{HGB}$',
    'gamma_HGB': '$\gamma_{HGB}$',
    'ktr_HGB': '$k_{trHGB}$',
    'slopeA_HGB': '$slope_{AHGB}$',
    'slopeD_HGB': '$slope_{DHGB}$',
    'B_PLT': '$B_{PLT}$',
    'gamma_PLT': '$\gamma_{PLT}$',
    'ktr_PLT': '$k_{trPLT}$',
    'slopeA_PLT': '$slope_{APLT}$',
    'slopeD_PLT': '$slope_{DPLT}$',
    'B_WBC': '$B_{WBC}$',
    'gamma_WBC': '$\gamma_{WBC}$',
    'ktr_WBC': '$k_{trWBC}$',
    'slopeA_WBC': '$slope_{AWBC}$',
    'slopeD_WBC': '$slope_{DWBC}$',
    'B_NEUT': '$B_{NEUT}$',
    'gamma_NEUT': '$\gamma_{NEUT}$',
    'ktr_NEUT': '$k_{trNEUT}$',
    'slopeA_NEUT': '$slope_{ANEUT}$',
    'slopeD_NEUT': '$slope_{DNEUT}$'
}

cell_type_names = {
    'HGB': 'Duration of anemia',
    'WBC': 'Duration of leukopenia',
    'PLT':'Duration of thrombocytopenic crisis',
    'NEUT':'Duration of granulocytopenia',
}
# Loop through the significant correlations to create subplots
for i, (idx, row) in enumerate(significant_correlations.iterrows(), start=1):
    plt.subplot(num_rows, num_cols, i)
    sns.scatterplot(data=data, x=row['Cell Type'], y=row['Feature'])
    plt.xlabel(cell_type_names[row['Cell Type']])
    plt.ylabel(latex_feature_names[row['Feature']])

# Adjust the layout
plt.tight_layout()
# Save the figure
# plt.savefig('D:/OneDrive/ModelParameterClassification/Results/significant_correlations.png', dpi=300)
plt.show()