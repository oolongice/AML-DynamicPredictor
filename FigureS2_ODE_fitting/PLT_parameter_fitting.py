import os
import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
from pyswarm import pso
import re
import csv
from tqdm import tqdm

# 读取血常规数据文件
# data = "Cleaned_Blood_routine_data.xlsx"  # 请使用您的CSV文件路径替换
# data1 = pd.read_excel(data, engine='openpyxl')
data1 = "../data/combined_Blood_routine_data.csv"
data1 = pd.read_csv(data1, sep=',')

# 读取 用药方案 文件
# file_path = "updated_drug_dosage_plan.xlsx"  # 请使用您的文件路径替换
# data2 = pd.read_excel(file_path, engine='openpyxl')
data2 = "../data/combined_drug_dosage_plan.csv"
data2 = pd.read_csv(data2, sep=',')

# 创建用于存储图像的文件夹
output_dir = "Results/Figures_PLT"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def analyze_patient(patient_id, data1, data2):
    # 筛选出病人ID为patient_id的数据
    patient_data1 = data1[data1['id'] == patient_id]
    # 筛选 Start Time 小于等于 25 的数据行
    patient_data2 = data2[(data2['id'] == patient_id) & (data2['Start Time'] <= 25)]

    # 从筛选出的数据中提取'time'和'PLT'两列
    time_PLT_data = patient_data1[['time', 'PLT']]
    # 将DataFrame转换为NumPy数组
    time_PLT_array = time_PLT_data.to_numpy()
    timePLT = time_PLT_array[:, 0]
    xma_data = time_PLT_array[:, 1]
    # 筛选出时间小于等于0的数据
    time_le_0_data = time_PLT_data[time_PLT_data['time'] <= 0]
    # 检查是否存在 time 为 0 的数据点
    time_0_data = time_PLT_data[time_PLT_data['time'] == 0]

    if not time_0_data.empty:
        # 如果存在 time 为 0 的数据点，提取对应的 PLT 值并赋给 B
        B0 = time_0_data['PLT'].values[0]
    elif not time_le_0_data.empty:
        # 如果不存在 time 为 0 的数据点，但存在 time <= 0 的数据点，
        # 找到 time 最接近 0 的前后两个数据点
        time_before_0 = time_le_0_data.loc[time_le_0_data['time'].idxmax()]
        # Find the data point just after time = 0
        time_after_0 = time_PLT_data[time_PLT_data['time'] > 0].iloc[0]

        # 对它们的 PLT 值进行插值，得到 time 为 0 时刻的 PLT 值，并赋给 B
        B0 = np.interp(0, [time_before_0['time'], time_after_0['time']], [time_before_0['PLT'], time_after_0['PLT']])
    else:
        # 如果不存在 time <= 0 的数据点，找到 time 最接近且大于0的数据点
        time_after_0 = time_PLT_data.loc[time_PLT_data['time'].idxmin()]
        B0 = time_after_0['PLT']

    kma = 2.3765

    patient_data2 = data2[data2['id'] == patient_id]

    def preprocess_dosage_data(patient_data2):
        from collections import defaultdict
        # 初始化药物的剂量字典，其中键是(药物, 时间点)元组，值是累积剂量
        dosage_timeline = defaultdict(float)

        # 遍历给药记录，累积每天的剂量
        for _, row in patient_data2.iterrows():
            drug = row['Drug']
            start_time = row['Start Time']
            end_time = min(row['End Time'], 25)  # 如果End Time大于25，将其设置为25
            dosage_str = row['Dosage']

            # 使用正则表达式匹配剂量字符串中的数字，包括可能的小数点和空格
            match = re.search(r'(\d+\.?\d*)\s*mg', dosage_str)
            if match:
                # 转换剂量为浮点数
                dosage_value = float(match.group(1))
                # 将每个时间点的用药量累加
                for time_point in range(start_time, end_time + 1):
                    key = (drug, time_point)  # 使用药物和时间点作为键
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

    # 使用此函数预处理数据
    dosage_timeline = preprocess_dosage_data(patient_data2)

    # 定义时间范围
    t_start = 0
    t_end = 30  # 35
    num_points = 100
    t_eval = np.linspace(t_start, t_end, num_points)

    def ode_system(x, t, p, get_u):
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

    def objective(p):
        # 使用 p 中的参数
        B, gamma, ktr, slopeA, slopeD = p
        Bbm = B * kma / ktr

        # 更新 x0
        x0_updated = [Bbm, Bbm, Bbm, Bbm, B0]

        # 使用 solve_ivp 求解 ODE 系统
        solution = solve_ivp(lambda t, x: ode_system(x, t, p, get_u), (t_start, t_end), x0_updated, method='RK45')

        # 提取我们关心的变量（例如 xma）
        xma_values_at_timePLT = np.interp(timePLT, solution.t, solution.y[-1])

        # # 计算相对误差
        # relative_error = np.sum(((xma_values_at_timePLT - xma_data) / np.maximum(np.abs(xma_data), 1e-8)) ** 2)
        # # 计算归一化的均方误差（NMSE）
        # nmse = relative_error / len(xma_data)

        # # 计算Root Mean Squared Error均方根误差
        rmse = np.sqrt(np.mean((xma_values_at_timePLT - xma_data) ** 2))
        # # 计算NRMSE(Normalized Root Mean Squared Error)
        # y_max = np.max(xma_data)
        # y_min = np.min(xma_data)
        # nrmse = rmse / (y_max - y_min)

        # 定义正则化系数
        reg_coeff = 0.0001
        # 计算正则化项
        reg_term = reg_coeff * np.sum(np.square(p))

        # 目标函数包括相对误差和正则化项
        objective_value = rmse + reg_term
        # objective_value = nrmse + reg_term

        return objective_value

    # 参数边界
    lb = [20, 0.1, 0.1, 1e-3, 1e-3]  # 每个参数的下界   B, gamma, ktr, slopeA, slopeD = p
    ub = [350, 1.5, 1.5, 15, 15]

    swarmsize = 100
    maxiter = 150
    # 使用 PSO 进行参数优化
    popt, fopt = pso(objective, lb, ub, swarmsize=swarmsize, maxiter=maxiter)

    # 输出优化后的参数
    B_opt, gamma_opt, ktr_opt, slopeA_opt, slopeD_opt = popt

    # 更新 x0
    Bbm_opt = B_opt * kma / ktr_opt
    x0_opt = [Bbm_opt, Bbm_opt, Bbm_opt, Bbm_opt, B0]

    # 使用优化后的参数求解ODE
    p_opt = [B_opt, gamma_opt, ktr_opt, slopeA_opt, slopeD_opt]
    solution_opt = solve_ivp(lambda t, x: ode_system(x, t, p_opt, get_u), (t_start, t_end), x0_opt, t_eval=t_eval,
                             method='RK45')
    
    # 提取我们关心的变量（例如 xma）
    xma_values_at_timePLT_opt = np.interp(timePLT, solution_opt.t, solution_opt.y[-1])

    # Calculate R^2
    y_obs = xma_data  # Observed data
    y_pred = xma_values_at_timePLT_opt  # Predicted data
    ss_res = np.sum((y_obs - y_pred) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_obs - np.mean(y_obs)) ** 2)  # Total sum of squares
    r_squared = 1 - (ss_res / ss_tot)

    # 筛选出 time 大于等于 0 的数据点
    time_ge_0_data = time_PLT_data[time_PLT_data['time'] >= 0]

    # 提取 time 和 PLT 列的值
    time_ge_0 = time_ge_0_data['time'].values
    PLT_ge_0 = time_ge_0_data['PLT'].values

    plt.rcParams["font.weight"] = "bold"
    plt.rcParams['font.size'] = 10
    plt.rcParams["axes.labelweight"] = "bold"
    # 绘制 xma 随时间变化的图像
    plt.figure()
    plt.plot(solution_opt.t, solution_opt.y[-1], label='Numerical simulation')
    plt.scatter(time_ge_0, PLT_ge_0, label='Experimental data')
    plt.xlabel('Time (days)')
    plt.ylabel('PLT count ($10^9/L$)')
    plt.legend(frameon=False)
    plt.title(f"Patient ID: {patient_id}", fontsize=10, fontweight='bold')
    # 保存图像到文件夹中
    figure_path = os.path.join(output_dir, f'patient_{patient_id}.png')
    plt.savefig(figure_path, dpi=300)
    plt.close()  # 关闭图形，避免内存泄漏

    return patient_id, B_opt, gamma_opt, ktr_opt, slopeA_opt, slopeD_opt, fopt, r_squared


def analyze_and_optimize(patient_id):
    # 调用 analyze_patient 函数，传入共享的 data1 和 data2 数据
    return analyze_patient(patient_id, data1, data2)


def main():
    # 设置无头模式以防止图表显示
    os.environ['MPLBACKEND'] = 'Agg'

    # 获取 data1 中所有病人的 ID
    patient_ids = data1['id'].astype(int).unique()
    # patient_ids = new_data['id'].unique()
    # patient_ids = [18511539,18609922]

    # 使用 multiprocessing.Pool 来并行处理
    with mp.Pool(processes=mp.cpu_count()-1) as pool:  # 限制最大进程数为90   服务器核心为96
        # results = pool.map(analyze_and_optimize, patient_ids)
        # 使用 tqdm 创建进度条
        results = []
        for result in tqdm(pool.imap_unordered(analyze_and_optimize, patient_ids), total=len(patient_ids)):
            results.append(result)

    # 将结果写入CSV文件
    with open('optimized_parameters_and_R2_PLT.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['patient_id', 'B_opt', 'gamma_opt', 'ktr_opt', 'slopeA_opt','slopeD_opt', 'error', 'r_squared'])
        # 写入每个病人的结果
        for result in results:
            writer.writerow(result)


if __name__ == '__main__':
    main()




