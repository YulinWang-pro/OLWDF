import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from onlinelearning.ftrl_adp import *
from sklearn import svm
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
import pandas as pd
import math
def calculate_metrics(y_true, y_pred,y_pred_proba=None):
    """
    计算二分类任务的Recall、Precision、F1分数
    参数:
        y_true: 真实标签列表（0或1）
        y_pred: 预测标签列表（0或1）
    返回:
        recall, precision, f1
    """
    # 转换为numpy数组以提高计算效率
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算混淆矩阵
    tp = np.sum((y_true == 1) & (y_pred == 1))  # 真正例
    fp = np.sum((y_true == 0) & (y_pred == 1))  # 假正例
    fn = np.sum((y_true == 1) & (y_pred == 0))  # 假反例

    # 计算指标（避免除以零）
    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    # 计算AUC
    if y_pred_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            # 处理只有一类标签的情况
            auc = np.nan
    else:
        auc = np.nan  # 如果没有概率值，AUC为NaN
    return recall, precision, f1,auc
def svm_classifier(train_x, train_y, test_x, test_y):
    best_score = 0
    best_C = -1
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        clf = svm.LinearSVC(C = C, max_iter = 100000)
        clf.fit(train_x,train_y)
        score = clf.score(test_x, test_y)
        if score > best_score:
            best_score = score
            best_C = C
    return  best_score, best_C

def calculate_svm_error(X_input, Y_label,n):
    length = int(0.7*n)
    X_train = X_input[:length, :]
    Y_train = Y_label[:length]
    X_test = X_input[length:, :]
    Y_test = Y_label[length:]
    best_score, best_C = svm_classifier(X_train, Y_train, X_test, Y_test)
    error = 1.0 - best_score
    return error, best_C

def generate_Xmask(n, X_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        if row in Y_label_masked:
            x = X_input[row]
            y = Y_label[row]
            y_not = 100
            p, w = classifier.fit(indices, x, y_not, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            predict.append(p)
        else:
            x = X_input[row]
            y = Y_label[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            decays.append(decay)
            predict.append(p)

    X_Zero_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    # svm_error, _ = calculate_svm_error(X_input[:, 1:], Y_label, n)

    return X_Zero_CER#, svm_error

def generate_Xmask_trap(n, X_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        if row in Y_label_masked:
            x = np.array(X_input[row]).data
            y = Y_label[row]
            y_not = 100
            p, w = classifier.fit(indices, x, y_not, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            predict.append(p)
        else:
            x = X_input[row]
            y = Y_label[row]
            p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
            error = [int(np.abs(y - p) > 0.5)]

            errors.append(error)
            decays.append(decay)
            predict.append(p)

    X_Zero_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    # svm_error, _ = calculate_svm_error(X_input[:, 1:], Y_label, n)

    return X_Zero_CER#, svm_error

def generate_X_delay(n,X_input,Y_true,Y_scenario,pseudo_labels,delay_rounds,decay_choice,contribute_error_rate):
    np.random.seed(SEED)

    """填充"""
    # combined_labels_ensemble = Y_scenario.copy()
    # combined_labels_X = Y_scenario.copy()
    # combined_labels_Z = Y_scenario.copy()
    # pseudo_mask = np.isnan(combined_labels_ensemble) & ~np.isnan(pseudo_labels_ensemble)
    # combined_labels_ensemble[pseudo_mask] = pseudo_labels_ensemble[pseudo_mask]
    # pseudo_mask_1 = np.isnan(combined_labels_X) & ~np.isnan(pseudo_labels_x)
    # combined_labels_X[pseudo_mask_1] = pseudo_labels_x[pseudo_mask_1]
    # pseudo_mask_2 = np.isnan(combined_labels_Z) & ~np.isnan(pseudo_labels_z)
    # combined_labels_Z[pseudo_mask_2] = pseudo_labels_z[pseudo_mask_2]



    combined_labels_X = Y_scenario.copy()
    pseudo_mask_1 = np.isnan(combined_labels_X) & ~np.isnan(pseudo_labels)
    combined_labels_X[pseudo_mask_1] = pseudo_labels[pseudo_mask_1]
    Y_scenario=combined_labels_X

    # 初始化变量
    all_predictions = [None] * n
    all_pred_labels = [None] * n
    all_errors = [None] * n
    update_count = 0

    classifier = FTRL_ADP(
        decay=1.0, L1=0., L2=0., LP=1.,
        adaptive=True, n_inputs=len(X_input[-1])
    )

    # 缓冲区：存储等待标签的预测信息
    prediction_buffer = {}

    for current_round in range(n):
        indices = list(range(len(X_input[current_round])))
        x = np.array(X_input[current_round]).data
        #x = X_input[current_round]

        # ===== 阶段1: 处理延迟标签更新 =====
        if current_round >= delay_rounds:
            delayed_label = Y_scenario[current_round]
            original_round = current_round - delay_rounds

            # 检查是否有可用的延迟标签且不是缺失的
            if (original_round >= 0 and not np.isnan(delayed_label) and original_round in prediction_buffer):
                historical_indices,historical_x, historical_p, historical_y_pred = prediction_buffer[original_round]

                # 使用延迟标签更新模型
                updated_p, decay, loss, w = classifier.fit(
                    historical_indices, historical_x, delayed_label, decay_choice, contribute_error_rate)
                update_count += 1

                # 从缓冲区移除已处理的预测
                del prediction_buffer[original_round]

        # ===== 阶段2: 当前轮预测 =====
        p = classifier.predict(indices, x)
        y_pred = 1 if p > 0.5 else 0

        # 存储预测结果
        all_predictions[current_round] = p
        all_pred_labels[current_round] = y_pred

        # 将当前预测存入缓冲区（用于延迟标签场景）
        if delay_rounds > 0:
            prediction_buffer[current_round] = (indices,x, p, y_pred)

        # ===== 计算错误率 =====
        if not np.isnan(Y_true[current_round]):#Y_scenario存在缺失的可能性，用Y_true
            error = int(np.abs(Y_true[current_round] - y_pred) > 0.5)
            all_errors[current_round] = error

    # ===== 计算累积错误率 - 只考虑有效样本 =====
    valid_errors = all_errors
    CER = np.cumsum(valid_errors) / (np.arange(len(valid_errors)) + 1.0)


    # ===== 计算性能指标 - 只考虑有效样本 =====
    valid_indices = [i for i in range(n)]

    assert  len(Y_true)==n

    if valid_indices:
        valid_true = [Y_true[i] for i in valid_indices]
        valid_pred = [all_pred_labels[i] for i in valid_indices]
        valid_pred_proba = [all_predictions[i] for i in valid_indices]
        recall, precision, f1, auc = calculate_metrics(valid_true, valid_pred, valid_pred_proba)
    else:
        recall, precision, f1, auc = 0.0, 0.0, 0.0, 0.0
    svm_error=None
    return CER, svm_error, recall, precision, f1, auc
def generate_X_Y(n, X_input, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_fill_x[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))

    X_Zero_CER  = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    svm_error,_ = calculate_svm_error(X_input[:,1:], Y_label, n)

    return X_Zero_CER, svm_error

def generate_X_Y_trap(n, X_input, Y_label_fill_x, Y_label, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label_fill_x[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))

    X_Zero_CER  = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return X_Zero_CER

def generate_Z(n, X_input, Y_label_fill_z, Y_label, decay_choice, contribute_error_rate):
    errors  = []
    decays  = []
    predict = []
    mse     = []

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_fill_z[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)
        predict.append(p)
        mse.append(mean_squared_error(predict[:row+1], Y_label[:row+1]))

    X_Zero_CER  = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)
    svm_error,_ = calculate_svm_error(X_input[:,1:], Y_label, n)

    return X_Zero_CER, svm_error

def generate_cap(n, X_input, Y_label, decay_choice, contribute_error_rate):
    errors=[]
    decays=[]

    classifier = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])

    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row].data
        y = Y_label[row]
        p, decay,loss,w= classifier.fit(indices, x, y ,decay_choice,contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]

        errors.append(error)
        decays.append(decay)

    Z_imp_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return Z_imp_CER

def generate_tra(n, X_input, Y_label, decay_choice, contribute_error_rate):
    errors = []
    classifier = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(X_input[-1]))
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label[row]
        p, decay, loss, w = classifier.fit(indices, x, y, decay_choice, contribute_error_rate)
        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    imp_CER = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return imp_CER

def draw_cap_error_picture(ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER,
                           Z_impl_CER_fill, svm_error, dataset):
    n = len(ensemble_XZ_imp_CER)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER")  #
    x = range(n)

    Z_imp_CER_fill,  = plt.plot(x, ensemble_XZ_imp_CER_fill, color = 'green',    linestyle = "--")    # the error of z_imp
    X_Zero_CER_fill, = plt.plot(x, X_Zero_CER_fill,          color = 'blue',     linestyle = "-")      # the error of x_zero
    Z_imp_CER,       = plt.plot(x, ensemble_XZ_imp_CER,      color = 'magenta',  linestyle = "-.")    # the error of z_imp
    X_Zero_CER,      = plt.plot(x, X_Zero_CER,               color = 'black',    linestyle = ":")      # the error of x_zero
    Z_impl_CER,      = plt.plot(x, Z_impl_CER,               color = 'pink',     linestyle = "solid")      # the error of x_zero
    Z_impl_CER_fill, = plt.plot(x, Z_impl_CER_fill,          color = 'cyan')      # the error of x_zero

    svm_error, = plt.plot(x, [svm_error] * n ,color='red')   # the error of svm

    plt.legend(handles=[Z_imp_CER_fill,             X_Zero_CER_fill,   Z_imp_CER,             X_Zero_CER,   Z_impl_CER,  Z_impl_CER_fill,   svm_error],
               labels=["ensemble_XZ_imp_CER_fill", "X_Zero_CER_fill", "ensemble_XZ_imp_CER", "X_Zero_CER", "Z_impl_CER", "Z_impl_CER_fill", "svm_error"])

    plt.title(dataset + "_The Cumulative error rate(CER) of ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, "
                        "X_Zero_CER, Z_impl_CER, Z_impl_CER_fill, SVM_CER")
    plt.show()
    # plt.clf()

def draw_cap_error_picture_tra(ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER, Z_impl_CER_fill, dataset):
    n = len(ensemble_XZ_imp_CER)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER")
    x = range(n)

    Z_imp_CER_fill,      = plt.plot(x, ensemble_XZ_imp_CER_fill, color = 'green',    linestyle = "--")    # the error of z_imp
    X_Zero_CER_fill,     = plt.plot(x, X_Zero_CER_fill,          color = 'blue',     linestyle = "-")      # the error of x_zero
    ensemble_XZ_imp_CER, = plt.plot(x, ensemble_XZ_imp_CER,      color = 'magenta',  linestyle = "-.")    # the error of z_imp
    X_Zero_CER,          = plt.plot(x, X_Zero_CER,               color = 'black',    linestyle = ":")      # the error of x_zero
    Z_impl_CER,          = plt.plot(x, Z_impl_CER,               color = 'pink',     linestyle = "solid")      # the error of x_zero
    Z_impl_CER_fill,     = plt.plot(x, Z_impl_CER_fill,          color = 'cyan')      # the error of x_zero

    plt.legend(handles=[Z_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER, Z_impl_CER_fill],
               labels=["ensemble_XZ_imp_CER_fill", "X_Zero_CER_fill", "ensemble_XZ_imp_CER", "X_Zero_CER", "Z_impl_CER", "Z_impl_CER_fill"])

    plt.title(dataset + "_The Cumulative error rate(CER) of ensemble_XZ_imp_CER_fill, X_Zero_CER_fill, ensemble_XZ_imp_CER, X_Zero_CER, Z_impl_CER, Z_impl_CER_fill")
    plt.show()

def draw_cap_error_picture_1(Z_imp_CER, X_Zero_CER,):
    n = len(Z_imp_CER)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER")
    x = range(n)

    Z_imp_CER,  = plt.plot(x, Z_imp_CER  , color='green')
    X_Zero_CER, = plt.plot(x, X_Zero_CER, color='blue')

    plt.legend(handles=[Z_imp_CER, X_Zero_CER], labels=["Z_imp_CER", "X_Zero_CER"])

    plt.title("The Cumulative error rate(CER) of z_imp_CER, x_zero_CER")
    plt.show()
    plt.clf()

def draw_tra_error_picture(error_arr_Z, error_arr_X):
    n = len(error_arr_Z)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1.0))
    plt.xlim((0, n))
    plt.ylabel("CER")  #

    x = range(n)
    error_arr_Z, = plt.plot(x, error_arr_Z, color='green')
    error_arr_X, = plt.plot(x, error_arr_X, color='blue')

    plt.legend(handles=[error_arr_Z, error_arr_X], labels=["error_arr_Z", "error_arr_X"])

    plt.title("The CER of trapezoid data stream")
    plt.show()
    plt.clf()

def print_results_table(results_dict):
    """
    打印规范的性能指标表格
    """
    headers = ["Method", "Scenario", "CER", "Recall", "Precision", "F1", "AUC"]

    # 准备数据行
    rows = []

    for method_scenario, metrics in results_dict.items():
        # 解析方法名和场景
        parts = method_scenario.split('_')
        method = parts[0]
        scenario = "delay_missing" if "missing" in method_scenario else "delay"

        # 提取指标
        cer = metrics.get('CER', 'N/A')
        recall = metrics.get('recall', 'N/A')
        precision = metrics.get('precision', 'N/A')
        f1 = metrics.get('f1', 'N/A')
        auc = metrics.get('auc', 'N/A')

        # 添加行
        rows.append([
            method,
            scenario,
            f"{cer:.4f}" if isinstance(cer, (int, float)) else cer,
            f"{recall:.4f}" if isinstance(recall, (int, float)) else recall,
            f"{precision:.4f}" if isinstance(precision, (int, float)) else precision,
            f"{f1:.4f}" if isinstance(f1, (int, float)) else f1,
            f"{auc:.4f}" if isinstance(auc, (int, float)) else auc
        ])

    # 打印表格
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 80)
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print("=" * 80)

    # 可选：创建DataFrame以便进一步处理
    df = pd.DataFrame(rows, columns=headers)
    return df


def draw_cap_error_picture_extended(Z_imp_CER_delay, X_Zero_CER_delay, Z_imp_CER_delay_missing,
                                    X_Zero_CER_delay_missing, svm_error_delay=None, svm_error_delay_missing=None,
                                    y_imp_delay=None, y_imp_delay_missing=None):
    """
    绘制扩展的CER对比图 - 仅显示延迟+缺失场景
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # 获取n值（使用延迟+缺失场景的数据长度）
    n = 0
    if Z_imp_CER_delay_missing is not None:
        n = len(Z_imp_CER_delay_missing)
    elif X_Zero_CER_delay_missing is not None:
        n = len(X_Zero_CER_delay_missing)
    elif y_imp_delay_missing is not None:
        n = len(y_imp_delay_missing)
    else:
        n = 100  # 默认值

    ax.set_ylim((0, 1))
    ax.set_xlim((0, n))
    ax.set_ylabel("CER")
    ax.set_xlabel("Time Step")
    ax.set_title("Delay + Missing Scenario")

    x = range(n)

    # 绘制Z_imp (Ensemble) - Delay + Missing
    if Z_imp_CER_delay_missing is not None:
        ax.plot(x, Z_imp_CER_delay_missing, color='green', label='Z_imp (Ensemble)')

    # 绘制X_Zero (Baseline) - Delay + Missing
    if X_Zero_CER_delay_missing is not None:
        ax.plot(x, X_Zero_CER_delay_missing, color='blue', label='X_Zero (Baseline)')

    # 绘制Y_imp (Online Imputation) - Delay + Missing
    if y_imp_delay_missing is not None:
        ax.plot(x, y_imp_delay_missing, color='orange', label='Y_imp (Online Imputation)')

    # 绘制SVM - Delay + Missing
    if svm_error_delay_missing is not None:
        if hasattr(svm_error_delay_missing, '__iter__'):
            # 如果是数组，直接绘制
            ax.plot(x, svm_error_delay_missing, color='red', linestyle='--', label='SVM (Batch)')
        else:
            # 如果是单个值，绘制水平线
            ax.plot(x, [svm_error_delay_missing] * n, color='red', linestyle='--', label='SVM (Batch)')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.clf()


