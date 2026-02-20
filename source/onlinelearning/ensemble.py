import numpy as np
from onlinelearning.ftrl_adp import FTRL_ADP
from sklearn.metrics import roc_auc_score
SEED=42
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
def ensemble_delay(n,X_input,Z_input,Y_label,Y_scenario,pseudo_labels,delay_rounds,decay_choice,contribute_error_rate):

    combined_labels_ensemble = Y_scenario.copy()
    pseudo_mask = np.isnan(combined_labels_ensemble) & ~np.isnan(pseudo_labels)
    combined_labels_ensemble[pseudo_mask] = pseudo_labels[pseudo_mask]
    Y_scenario=combined_labels_ensemble

    np.random.seed(SEED)
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])
    all_predictions = [None] * n
    all_pred_labels = [None] * n
    all_errors = [None] * n
    x_loss=0
    z_loss=0
    lamda=0.5
    eta = 0.001
    # 缓冲区：存储等待标签的预测信息
    prediction_buffer = {}

    for current_round in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[current_round]
        z = Z_input[current_round]

        if current_round >= delay_rounds:
            delayed_label = Y_scenario[current_round]
            original_round = current_round - delay_rounds
            # 检查是否有可用的延迟标签且不是缺失的
            if (original_round >= 0 and not np.isnan(delayed_label) and original_round in prediction_buffer):
                buffer_data = prediction_buffer[original_round]
                historical_indices_O,historical_indices_Z,historical_x_zero, historical_x_z_imp, historical_p_O, historical_p_Z, historical_p_ensemble, historical_y_pred_ensemble = buffer_data

                # 更新两个基分类器
                updated_p_O, decay_O, loss_x, w_O = classifier_X.fit(
                    historical_indices_O, historical_x_zero, delayed_label, decay_choice, contribute_error_rate
                )

                updated_p_Z, decay_Z, loss_z, w_Z = classifier_Z.fit(
                    historical_indices_Z, historical_x_z_imp, delayed_label, decay_choice, contribute_error_rate
                )
                x_loss += loss_x
                z_loss += loss_z
                lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

                del prediction_buffer[original_round]

        p_O, w_O = classifier_X.fit(indices, x, 100, decay_choice, contribute_error_rate)
        p_Z, w_Z = classifier_Z.fit(indices, z, 100, decay_choice, contribute_error_rate)

        p_ensemble = sigmoid(lamda * np.dot(w_O, x) + ( 1.0 - lamda ) * np.dot(w_Z, z))
        y_pred_ensemble = 1 if p_ensemble > 0.5 else 0

        # 存储预测结果
        all_predictions[current_round] = p_ensemble
        all_pred_labels[current_round] = y_pred_ensemble
        # 将当前预测存入缓冲区（用于延迟标签场景）
        if delay_rounds > 0:
            prediction_buffer[current_round] = (indices,indices,x, z, p_O, p_Z, p_ensemble, y_pred_ensemble)

        # ===== 计算错误率 =====
        if not np.isnan(Y_label[current_round]):
            error = int(np.abs(Y_label[current_round] - y_pred_ensemble) > 0.5)
            all_errors[current_round] = error

    # ===== 计算累积错误率 - 只考虑有效样本 =====
    valid_errors = all_errors
    CER = np.cumsum(valid_errors) / (np.arange(len(valid_errors)) + 1.0)

    # ===== 计算性能指标 - 只考虑有效样本 =====
    valid_indices = [i for i in range(n)]
    if valid_indices:
        valid_true = [Y_label[i] for i in valid_indices]
        valid_pred = [all_pred_labels[i] for i in valid_indices]
        valid_pred_proba = [all_predictions[i] for i in valid_indices]
        recall, precision, f1, auc = calculate_metrics(valid_true, valid_pred, valid_pred_proba)
    else:
        recall, precision, f1, auc = 0.0, 0.0, 0.0, 0.0

    return CER, recall, precision, f1, auc

def tra_ensemble_delayed(n,X_input,Z_input,Y_label,Y_scenario,pseudo_labels,delay_rounds,decay_choice,contribute_error_rate):
    """
    统一的延迟标签集成学习函数 - 同时支持延迟和缺失标签
    评估样本数 = 总样本数 - 延迟步数，确保公平性
    """

    combined_labels_ensemble = Y_scenario.copy()
    pseudo_mask = np.isnan(combined_labels_ensemble) & ~np.isnan(pseudo_labels)
    combined_labels_ensemble[pseudo_mask] = pseudo_labels[pseudo_mask]
    Y_scenario=combined_labels_ensemble

    np.random.seed(SEED)

    # 初始化变量
    all_predictions = [None] * n
    all_pred_labels = [None] * n
    all_errors = [None] * n
    update_count = 0
    alpha_history = []

    # 初始化分类器
    classifier_O = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(X_input[-1]))
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=len(Z_input[-1]))

    # 初始化集成权重
    alpha = 0.5
    R_O = 0
    R_Z = 0
    eta = 0.001
    # 缓冲区：存储等待标签的预测信息
    prediction_buffer = {}

    for current_round in range(n):


        indices_O = list(range(len(X_input[current_round])))
        indices_Z = list(range(len(Z_input[current_round])))

        x_zero = np.array(X_input[current_round]).data
        x_z_imp = np.array(Z_input[current_round]).data
        # x_zero = X_input_zero[current_round]
        # x_z_imp = X_input_z_imp[current_round]

        # ===== 阶段1: 处理延迟标签更新 =====
        if current_round >= delay_rounds:
            delayed_label = Y_scenario[current_round]
            original_round = current_round - delay_rounds

            # 检查是否有可用的延迟标签且不是缺失的
            if (original_round >= 0 and not np.isnan(delayed_label) and original_round in prediction_buffer):
                buffer_data = prediction_buffer[original_round]
                historical_indices_O,historical_indices_Z,historical_x_zero, historical_x_z_imp, historical_p_O, historical_p_Z, historical_p_ensemble, historical_y_pred_ensemble = buffer_data

                # 更新两个基分类器
                updated_p_O, decay_O, loss_O, w_O = classifier_O.fit(
                    historical_indices_O, historical_x_zero, delayed_label, decay_choice, contribute_error_rate,
                )

                updated_p_Z, decay_Z, loss_Z, w_Z = classifier_Z.fit(
                    historical_indices_Z, historical_x_z_imp, delayed_label, decay_choice, contribute_error_rate,
                )

                # 更新集成权重 - 使用最初的方法
                R_O += loss_O
                R_Z += loss_Z
                alpha = np.exp(-eta * R_O) / (np.exp(-eta * R_O) + np.exp(-eta * R_Z))

                update_count += 1

                # 从缓冲区移除已处理的预测
                del prediction_buffer[original_round]

        p_O, w_O = classifier_O.fit(indices_O, x_zero, 100, decay_choice, contribute_error_rate)
        p_Z, w_Z = classifier_Z.fit(indices_Z, x_z_imp, 100, decay_choice, contribute_error_rate)
        p_ensemble = sigmoid(alpha * np.dot(w_O, x_zero) + ( 1.0 - alpha ) * np.dot(w_Z, x_z_imp))
        y_pred_ensemble = 1 if p_ensemble > 0.5 else 0

        # 存储预测结果
        all_predictions[current_round] = p_ensemble
        all_pred_labels[current_round] = y_pred_ensemble
        alpha_history.append(alpha)

        # 将当前预测存入缓冲区（用于延迟标签场景）
        if delay_rounds > 0:
            prediction_buffer[current_round] = (indices_O,indices_Z,x_zero, x_z_imp, p_O, p_Z, p_ensemble, y_pred_ensemble)

        # ===== 计算错误率 =====
        if not np.isnan(Y_label[current_round]):
            error = int(np.abs(Y_label[current_round] - y_pred_ensemble) > 0.5)
            all_errors[current_round] = error

    # ===== 计算累积错误率 - 只考虑有效样本 =====
    valid_errors = all_errors
    CER = np.cumsum(valid_errors) / (np.arange(len(valid_errors)) + 1.0)

    # ===== 计算性能指标 - 只考虑有效样本 =====
    valid_indices = [i for i in range(n)]
    if valid_indices:
        valid_true = [Y_label[i] for i in valid_indices]
        valid_pred = [all_pred_labels[i] for i in valid_indices]
        valid_pred_proba = [all_predictions[i] for i in valid_indices]
        recall, precision, f1, auc = calculate_metrics(valid_true, valid_pred, valid_pred_proba)
    else:
        recall, precision, f1, auc = 0.0, 0.0, 0.0, 0.0

    available_count = n - np.sum(np.isnan(Y_scenario))

    return CER, recall, precision, f1, auc

def ensemble(n, X_input, Z_input, Y_label , decay_choice, contribute_error_rate):
    errors=[]
    lamda_array = []

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        if np.isnan(Y_label[row]):
            continue
        else:
            x = X_input[row]
            y = Y_label[row]
            p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)

            z = Z_input[row]
            p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x,x) + ( 1.0 - lamda ) * np.dot(w_z,z))

            x_loss += loss_x
            z_loss += loss_z
            lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

            lamda_array.append(lamda)

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)
    lamda_array.savetxt()
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error

def ensemble_Xmask(n, X_input, Z_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    predict_x = []
    predict_y = []
    lamda_array = []
    errors=[]

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(Z_input[-1]))

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices_x = [i for i in range(len(X_input[row]))]
        indices_z = [i for i in range(len(Z_input[row]))]
        if row in Y_label_masked:
            x = X_input[row]
            z = Z_input[row]
            y = Y_label[row]
            y_not = 100  # 停止更新，返回两个数值（一定要注意这个位置）
            p_x, w_x = classifier_X.fit(indices_x, x, y_not, decay_choice, contribute_error_rate)
            p_z, w_z = classifier_Z.fit(indices_z, z, y_not, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)

        else:
            # 进行更新，并且会更新lambda
            x = X_input[row]
            y = Y_label[row]
            p_x, decay_x, loss_x, w_x = classifier_X.fit(indices_x, x, y ,decay_choice,contribute_error_rate)

            z = Z_input[row]
            p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices_z, z, y, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

            x_loss += loss_x
            z_loss += loss_z
            lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error

def ensemble_Xmask_trap(n, X_input, Z_input, Y_label, Y_label_masked, decay_choice, contribute_error_rate):
    predict_x = []
    predict_y = []
    errors=[]

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(len(X_input[row]))]
        if row in Y_label_masked:
            x = np.array(X_input[row]).data
            z = np.array(Z_input[row]).data
            y = Y_label[row]
            y_not = 100
            p_x, w_x = classifier_X.fit(indices, x, y_not, decay_choice, contribute_error_rate)
            p_z, w_z = classifier_Z.fit(indices, z, y_not, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)

        else:
            x = X_input[row]
            y = Y_label[row]
            p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)

            z = Z_input[row]
            p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

            p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

            x_loss += loss_x
            z_loss += loss_z
            lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

            error = [int(np.abs(y - p) > 0.5)]
            errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error


def ensemble_Y(n, X_input, Z_input, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate):
    errors=[]
    lamda_array = []

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = X_input.shape[1])
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_fill_x[row]
        p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)

        z = Z_input[row]
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

        p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

        x_loss += loss_x
        z_loss += loss_z
        lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))
        lamda_array.append(lamda)

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error, lamda_array

def ensemble_Y_trap(n, X_input, Z_input, Y_label, Y_label_fill_x, decay_choice, contribute_error_rate):
    errors=[]
    lamda_array = []

    classifier_X = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(X_input[-1]))
    classifier_Z = FTRL_ADP(decay = 1.0, L1 = 0., L2 = 0., LP = 1., adaptive = True, n_inputs = len(Z_input[-1]))

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001
    for row in range(n):
        indices_x = [i for i in range(len(X_input[row]))]
        indeces_z = [i for i in range(len(Z_input[row]))]
        x = np.array(X_input[row]).data
        y = Y_label_fill_x[row]
        p_x, decay_x, loss_x, w_x = classifier_X.fit(indices_x, x, y ,decay_choice,contribute_error_rate)

        z = np.array(Z_input[row]).data
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indeces_z, z, y, decay_choice, contribute_error_rate)

        p = sigmoid(lamda * np.dot(w_x, x) + ( 1.0 - lamda ) * np.dot(w_z, z))

        x_loss += loss_x
        z_loss += loss_z
        lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))
        lamda_array.append(lamda)

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)
    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)

    return ensemble_error, lamda_array

def logistic_loss(p,y):
    return (1 / np.log(2.0)) * (-y * np.log(p) - (1 - y) * np.log(1 - p))

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))