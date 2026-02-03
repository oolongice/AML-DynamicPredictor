#%%
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from scipy.integrate import solve_ivp

def preprocess_dosage_data(patient_data2):
    dosage_timeline = defaultdict(float)
    for _, row in patient_data2.iterrows():
        drug = row['Drug']
        start_time = row['Start Time']
        end_time = min(row['End Time'], 25)
        dosage_str = row['Dosage']
        match = re.search(r'(\d+\.?\d*)\s*mg', dosage_str)
        if match:
            dosage_value = float(match.group(1))
            for time_point in range(start_time, end_time + 1):
                key = (drug, time_point)
                dosage_timeline[key] += dosage_value
    return dosage_timeline

def get_u(t, dosage_timeline):
    uA, uD = 0, 0
    # 遍历dosage_timeline累积特定时间点的剂量
    for (drug, time_point), dosage in dosage_timeline.items():
        if time_point == t:
            if drug == 'Ara-C':
                uA += dosage * 2
            elif drug == 'Daunorubicin':
                uD += dosage
    return uA, uD
################################################################################

data1 = "../data/combined_Blood_routine_data.csv"
data1 = pd.read_csv(data1, sep=',')
data2 = "../data/combined_drug_dosage_plan.csv"
data2 = pd.read_csv(data2, sep=',')

patient_ids = data1['id'].astype(int).unique()

#%%
# HGB 
parameters = "../FigureS2_ODE_fitting/optimized_parameters_and_R2_HGB.csv"
parameters = pd.read_csv(parameters)
results_list = []
def HGB_ode_system(x, t, p, get_u, dosage_timeline):
    xpr, xtr1, xtr2, xtr3, xma = x
    B, gamma, ktr, slopeA, slopeD = p
    MMArac = 243.217
    MMDaun = 527.52  #543.519
    Vc = 37.33
    BSA = 1.78  # 范围[1.61, 2.07]
    dur = 1  # day
    CvA = 1 / (Vc * MMArac)
    CvD = 1 / (Vc * MMDaun)
    uA, uD = get_u(t, dosage_timeline)
    x1A = uA * BSA / dur
    x1D = uD * BSA / dur
    EA = slopeA * x1A * CvA
    ED = slopeD * x1D * CvD
    E = EA + ED
    dur = 1  # day
    ratio = max(B / max(xma, 1e-8), 0)
    dxpr_dt = ktr * xpr * (1 - E) * ratio ** gamma - ktr * xpr
    dxtr1_dt = ktr * (xpr - xtr1)
    dxtr2_dt = ktr * (xtr1 - xtr2)
    dxtr3_dt = ktr * (xtr2 - xtr3)
    dxma_dt = ktr * xtr3 - kma * xma
    return [dxpr_dt, dxtr1_dt, dxtr2_dt, dxtr3_dt, dxma_dt]

for patient_id in patient_ids:
    patient_data1 = data1[data1['id'] == patient_id]
    patient_data2 = data2[(data2['id'] == patient_id) & (data2['Start Time'] <= 25)]
    time_HGB_data = patient_data1[['time', 'HGB']]
    time_HGB_array = time_HGB_data.to_numpy()
    timeHGB = time_HGB_array[:, 0]
    xma_data = time_HGB_array[:, 1]
    time_le_0_data = time_HGB_data[time_HGB_data['time'] <= 0]
    time_0_data = time_HGB_data[time_HGB_data['time'] == 0]
    if not time_0_data.empty:
        B0 = time_0_data['HGB'].values[0]
    elif not time_le_0_data.empty:
        time_before_0 = time_le_0_data.loc[time_le_0_data['time'].idxmax()]
        time_after_0 = time_HGB_data[time_HGB_data['time'] > 0].iloc[0]
        B0 = np.interp(0, [time_before_0['time'], time_after_0['time']], [time_before_0['HGB'], time_after_0['HGB']])
    else:
        time_after_0 = time_HGB_data.loc[time_HGB_data['time'].idxmin()]
        B0 = time_after_0['HGB']

    kma = 2.3765

    patient_data2 = data2[data2['id'] == patient_id]
    dosage_timeline = preprocess_dosage_data(patient_data2)
    t_start = 0
    t_end = 30
    num_points = 100
    t_eval = np.linspace(t_start, t_end, num_points)

    row = parameters[parameters['patient_id'] == patient_id]
    if not row.empty:
        B = row['B_opt'].values[0]
        gamma = row['gamma_opt'].values[0]
        ktr = row['ktr_opt'].values[0]
        slopeA = row['slopeA_opt'].values[0]
        slopeD = row['slopeD_opt'].values[0]
        p = [B, gamma, ktr, slopeA, slopeD]
    Bbm = B * kma / ktr
    x0_updated = [Bbm, Bbm, Bbm, Bbm, B0]
    solution = solve_ivp(
        lambda t, x: HGB_ode_system(x, t, p, get_u, dosage_timeline), 
        (t_start, t_end), 
        x0_updated, 
        method='RK45'
    )
    xma_values_at_timeHGB = np.interp(timeHGB, solution.t, solution.y[-1])
    rmse = np.sqrt(np.mean((xma_values_at_timeHGB - xma_data) ** 2))
    
    y_max = np.max(xma_data)
    y_min = np.min(xma_data)
    # 防止除以0
    if y_max - y_min == 0:
        nrmse = 0
    else:
        nrmse = rmse / (y_max - y_min)
        
    reg_coeff = 0.0001
    reg_term = reg_coeff * np.sum(np.square(p))
    
    # 2. 将结果存入字典并添加到列表
    results_list.append({
        'patient_id': patient_id,
        'rmse': rmse,
        'nrmse': nrmse,
        'reg_term': reg_term
    })
HGB_error = pd.DataFrame(results_list)
save_path = "../data/all_hgb_nrmse.csv"
HGB_error.to_csv(save_path, index=False)
