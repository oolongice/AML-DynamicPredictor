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
# %%
# NEUT
parameters = "../FigureS2_ODE_fitting/optimized_parameters_and_R2_NEUT.csv"
parameters = pd.read_csv(parameters)
results_list = []
def NEUT_ode_system(x, t, p, get_u, dosage_timeline):
    # x 是一个包含 x1, x2, ..., xn 的向量
    # p 是一个包含参数的向量
    xpr, xtr1, xtr2, xtr3, xma = x
    B, gamma, ktr, slopeA, slopeD = p

    # 计算 Cv 值
    MMArac = 243.217
    MMDaun = 527.52  #543.519
    Vc = 37.33
    BSA = 1.78  # 范围[1.61, 2.07]
    dur = 1  # day
    CvA = 1 / (Vc * MMArac)
    CvD = 1 / (Vc * MMDaun)
    uA, uD = get_u(t, dosage_timeline)
    # 根据 uA 和 uD 分别计算 E 值
    x1A = uA * BSA / dur
    x1D = uD * BSA / dur
    EA = slopeA * x1A * CvA
    ED = slopeD * x1D * CvD

    # E 的总和是两种药物效果的总和
    E = EA + ED
    dur = 1  # day

    # 确保 B/xma 非负，避免无效的幂运算
    ratio = max(B / max(xma, 1e-8), 0)

    # 计算各个方程的导数
    dxpr_dt = ktr * xpr * (1 - E) * ratio ** gamma - ktr * xpr
    dxtr1_dt = ktr * (xpr - xtr1)
    dxtr2_dt = ktr * (xtr1 - xtr2)
    dxtr3_dt = ktr * (xtr2 - xtr3)
    dxma_dt = ktr * xtr3 - kma * xma

    # 返回导数向量
    return [dxpr_dt, dxtr1_dt, dxtr2_dt, dxtr3_dt, dxma_dt]

for patient_id in patient_ids:
    patient_data1 = data1[data1['id'] == patient_id]
    patient_data2 = data2[(data2['id'] == patient_id) & (data2['Start Time'] <= 25)]
    time_NEUT_data = patient_data1[['time', 'NEUT']]
    # 将DataFrame转换为NumPy数组
    time_NEUT_array = time_NEUT_data.to_numpy()
    timeNEUT = time_NEUT_array[:, 0]
    xma_data = time_NEUT_array[:, 1]
    # 筛选出时间小于等于0的数据
    time_le_0_data = time_NEUT_data[time_NEUT_data['time'] <= 0]
    # 检查是否存在 time 为 0 的数据点
    time_0_data = time_NEUT_data[time_NEUT_data['time'] == 0]

    if not time_0_data.empty:
        # 如果存在 time 为 0 的数据点，提取对应的 NEUT 值并赋给 B
        B0 = time_0_data['NEUT'].values[0]
    elif not time_le_0_data.empty:
        # 如果不存在 time 为 0 的数据点，但存在 time <= 0 的数据点，
        # 找到 time 最接近 0 的前后两个数据点
        time_before_0 = time_le_0_data.loc[time_le_0_data['time'].idxmax()]
        # Find the data point just after time = 0
        time_after_0 = time_NEUT_data[time_NEUT_data['time'] > 0].iloc[0]

        # 对它们的 NEUT 值进行插值，得到 time 为 0 时刻的 NEUT 值，并赋给 B
        B0 = np.interp(0, [time_before_0['time'], time_after_0['time']], [time_before_0['NEUT'], time_after_0['NEUT']])
    else:
        # 如果不存在 time <= 0 的数据点，找到 time 最接近且大于0的数据点
        time_after_0 = time_NEUT_data.loc[time_NEUT_data['time'].idxmin()]
        B0 = time_after_0['NEUT']
    kma = 2.3765
    patient_data2 = data2[data2['id'] == patient_id]
    dosage_timeline = preprocess_dosage_data(patient_data2)

    # 定义时间范围
    t_start = 0
    t_end = 30  # 35
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
        lambda t, x: NEUT_ode_system(x, t, p, get_u, dosage_timeline), 
        (t_start, t_end), 
        x0_updated, 
        method='RK45'
    )
    xma_values_at_timeNEUT = np.interp(timeNEUT, solution.t, solution.y[-1])
    rmse = np.sqrt(np.mean((xma_values_at_timeNEUT - xma_data) ** 2))
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
NEUT_error = pd.DataFrame(results_list)
save_path = "../data/all_neut_nrmse.csv"
NEUT_error.to_csv(save_path, index=False)
