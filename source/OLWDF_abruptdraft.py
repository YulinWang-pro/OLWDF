import warnings

import numpy as np
import copy
warnings.filterwarnings("ignore")
import sys
import math
import random
from time import time
sys.path.append("..")
from sklearn.svm import SVC
from evaluation.helpers import *
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from onlinelearning.ensemble import *
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from onlinelearning.online_learning import *
from semi_supervised.semiSupervised import *
from semi_supervised.semiSupervised_enhanced_v2 import *
from sklearn.preprocessing import StandardScaler
from em.online_expectation_maximization import OnlineExpectationMaximization
from onlinelearning.ftrl_adp_soft import *
from evaluation.drift_evaluation import *
from tool.result_recorder import *
SEED=2000
def create_delayed_labels_with_missing(Y_true, delay_rounds, missing_ratio=0.3, seed=SEED):
    np.random.seed(seed)
    n = len(Y_true)
    delayed_labels = np.full(n, np.nan)

    for t in range(n):
        if t + delay_rounds < n:
            #if np.random.rand() > missing_ratio or t < delay_rounds:
            if np.random.rand() > missing_ratio:
                delayed_labels[t + delay_rounds] = Y_true[t]
            else:
                delayed_labels[t + delay_rounds] = np.nan

    return delayed_labels


def create_missing_labels_only(Y_true, missing_ratio=0.3, seed=SEED):
    np.random.seed(seed)
    n = len(Y_true)
    missing_labels = np.full(n, np.nan)

    for t in range(n):
        if np.random.rand() > missing_ratio:
            missing_labels[t] = Y_true[t]
    return missing_labels


def create_delayed_labels_only(Y_true, delay_rounds, seed=SEED):
    np.random.seed(seed)
    n = len(Y_true)
    delayed_labels = np.full(n, np.nan)


    for t in range(n):
        if t + delay_rounds < n:
            delayed_labels[t + delay_rounds] = Y_true[t]

    return delayed_labels

if __name__ == '__main__':

    # [100, 150, 200, 250]
    delay_values = [50]
    WINDOW_SIZE_ZONE = 30
    DRIFT_START = int(2500)
    for DELAY_ROUNDS in delay_values:
        dataset = "RandomRBF_10_5k_abrupt_2500drift"

        # getting  hyperparameter
        contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change, decay_choice, shuffle = \
            get_my_cap_hyperparameter(dataset)
        MASK_NUM = 1
        X_target = pd.read_csv("../dataset/MaskData/" + dataset + "/X_process.txt", sep=" ", header=None)
        Y_label_target = pd.read_csv("../dataset/DataLabel/" + dataset + "/Y_label.txt", sep=' ', header=None)

        result = check_column_types_simple(X_target)
        if result == False:
            X_target = process_by_discreteness(X_target, method='auto')

        dataset="mixed"+dataset

        X_target = X_target.values
        Y_label_target = Y_label_target.values

        all_cont_indices = get_cont_indices(X_target)
        all_ord_indices = ~all_cont_indices

        n = X_target.shape[0]
        feat = X_target.shape[1]
        Y_label_target = Y_label_target.flatten()

        X=X_target.copy()
        Y_label=Y_label_target.copy()
        print(f"\n{'#' * 60}")
        print(f"Running Experiment with DELAY_ROUNDS = {DELAY_ROUNDS}")
        print(f"{'#' * 60}")
        POINT_WINDOW_SIZE = 50
        MISSING_RATIO = 0
        random.seed(SEED)
        np.random.seed(SEED)
        if shuffle == True:
            perm = np.arange(n)
            np.random.shuffle(perm)
            Y_label = Y_label[perm].copy()
            X = X[perm].copy()

        Y_full = Y_label.copy()
        Y_missing_only = create_missing_labels_only(Y_full, MISSING_RATIO,seed=SEED)
        Y_delayed_only = create_delayed_labels_only(Y_full, DELAY_ROUNDS,seed=SEED)
        Y_delayed_missing = create_delayed_labels_with_missing(Y_full, DELAY_ROUNDS, MISSING_RATIO,seed=SEED)

        random.seed(SEED)
        np.random.seed(SEED)
        X_masked = mask_types(X, MASK_NUM, seed=SEED)

        X_masked = np.array(X_masked)
        n=n-DELAY_ROUNDS
        X_masked=X_masked[:-DELAY_ROUNDS]



        #s etting hyperparameter
        max_iter = batch_size_denominator * 2
        BATCH_SIZE = math.ceil(n / batch_size_denominator)
        WINDOW_SIZE = math.ceil(n / window_size_denominator)
        NUM_ORD_UPDATES = 1
        batch_c = 8

        # start online copula imputation
        oem = OnlineExpectationMaximization(all_cont_indices, all_ord_indices, window_size=WINDOW_SIZE)
        j = 0
        X_imp_batch    = np.empty(X_masked.shape)
        Z_imp_batch    = np.empty(X_masked.shape)
        X_masked = np.array(X_masked)
        Y_label_fill_x = np.empty(Y_label.shape)
        Y_label_fill_z = np.empty(Y_label.shape)
        Y_label_fill_x_ensemble = np.empty(Y_label.shape)
        Y_label_fill_z_ensemble = np.empty(Y_label.shape)
        Y_label_fill_ensemble_hard = []
        Y_label_fill_ensemble_soft=[]


        processor_ensemble = SemiSupervisedEnhancedProcessor(
            delay_steps=DELAY_ROUNDS,
            buffer_size=len(X_masked[0]),
            lambda_decay=0.1,
            num_classes=2,
            fusion_method="fixed",
            view1_weight=0.5,
            view2_weight=0.5,
            confidence_threshold=0.6,
            #model="svc"
        )

        n_inputs = X_masked.shape[1] + 1
        ftrl_x_ensemble_soft = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=n_inputs)
        ftrl_z_ensemble_soft = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=n_inputs)

        buffer_feat_x = {}
        buffer_feat_z = {}
        origin_x_loss = 0
        origin_z_loss = 0
        x_loss = 0
        z_loss = 0
        x_soft_loss=0
        z_soft_loss=0
        x_hard_loss=0
        z_hard_loss=0
        origin_lamda = 0.5
        lamda = 0.5
        lamda_soft=0.5
        lamda_hard=0.5
        eta = 0.001
        results_lists = {
            'soft_X': {'preds': [], 'probs': [], 'errors': []},
            'soft_Z': {'preds': [], 'probs': [], 'errors': []},
            'soft_Ens': {'preds': [], 'probs': [], 'errors': []},
            'resnet_soft_X': {'preds': [], 'probs': [], 'errors': []},
            'resnet_soft_Z': {'preds': [], 'probs': [], 'errors': []},
            'resnet_soft_Ens': {'preds': [], 'probs': [], 'errors': []},
        }

        print(f"Start Online Loop for {n} rounds with Delay={DELAY_ROUNDS}...")
        loss_x_history = []
        loss_z_history = []
        for current_time in range(n):
            start = max(0, current_time - BATCH_SIZE + 1)
            end = current_time + 1
            X_batch = X_masked[start:end]
            batch_y_true=Y_label[start:end]
            if end < start:
                indices = np.concatenate((np.arange(end), np.arange(start, n, 1))) # 生成某个范围或区间对应的数组
            else:
                indices = np.arange(start, end, 1)
            if decay_coef_change == 1:
                this_decay_coef = batch_c / (j + batch_c)
            else:
                this_decay_coef = 0.5

            Z_imp, X_imp = oem.partial_fit_and_predict(X_batch, max_workers = 1, decay_coef = this_decay_coef)
            current_z_imp = Z_imp[-1]
            current_x_imp = X_imp[-1]
            Z_imp_batch[current_time, :] = current_z_imp
            X_imp_batch[current_time, :] = current_x_imp

            batch_y_scenario = Y_delayed_missing[DELAY_ROUNDS + start:end]
            labeled_indices = np.where(~np.isnan(batch_y_scenario))[0]
            unlabeled_indices = []
            for i in range(len(X_batch)):
                if i not in labeled_indices:
                    unlabeled_indices.append(i)
            percent = 5


            soft_labels_ensemble_all_hard=None
            soft_labels_ensemble_all_soft = None

            if len(X_batch)>2 and len(np.unique(batch_y_scenario[~np.isnan(batch_y_scenario)]))==2:
                train_x, label_train_x, initial_label_x = X_imp, batch_y_true,labeled_indices+1
                train_z, label_train_z, initial_label_z = Z_imp, batch_y_true, labeled_indices+1

                nneigh_x = DensityPeaks(train_x, percent)
                nneigh_z = DensityPeaks(train_z, percent)


                pseudo_batch_ensemble_hard, pseudo_batch_ensemble_soft = processor_ensemble.geometric_soft_label_propagation_ensemble(
                    unlabeled_indices, labeled_indices, nneigh_x.astype(int).flatten(), train_x,
                    train_z, nneigh_z.astype(int).flatten(),batch_y_true
                )
                Y_label_fill_ensemble_hard.append(pseudo_batch_ensemble_hard[-(DELAY_ROUNDS + 1)])
                Y_label_fill_ensemble_soft.append(pseudo_batch_ensemble_soft[-(DELAY_ROUNDS + 1)])

                soft_labels_ensemble_all_hard=pseudo_batch_ensemble_hard[-DELAY_ROUNDS:-1]
                soft_labels_ensemble_all_soft=pseudo_batch_ensemble_soft[-DELAY_ROUNDS:-1]
            else:
                Y_label_fill_x[indices[-1],]=np.nan
                Y_label_fill_z[indices[-1],]=np.nan
                Y_label_fill_x_ensemble[indices[-1],]=np.nan
                Y_label_fill_z_ensemble[indices[-1],] = np.nan
                Y_label_fill_ensemble_hard.append(np.array([np.nan]))
                Y_label_fill_ensemble_soft.append(np.array([np.nan]))


            row_x = X_masked[current_time].copy()
            row_x[np.isnan(row_x)] = 0
            feat_x_curr = np.insert(row_x, 0, 1.0)

            feat_z_curr = np.insert(current_z_imp, 0, 1.0)

            feat_indices = list(range(len(feat_x_curr)))
            update_idx = current_time - DELAY_ROUNDS

            if update_idx >= 0:
                real_delayed_label = Y_delayed_missing[current_time]

                real_delayed_Softlabel = np.zeros(2)
                real_delayed_Softlabel[int(real_delayed_label)] = 1.0
                pseudo_Softlabel_ensemble_hard=Y_label_fill_ensemble_hard[current_time]
                pseudo_Softlabel_ensemble_soft=Y_label_fill_ensemble_soft[current_time]


                combined_Softlabel_ensemble_soft = real_delayed_Softlabel
                if np.isnan(combined_Softlabel_ensemble_soft[0]) and not np.isnan(pseudo_Softlabel_ensemble_soft[0]):
                    combined_Softlabel_ensemble_soft = pseudo_Softlabel_ensemble_soft

                if not np.isnan(combined_Softlabel_ensemble_soft[0]) and update_idx in buffer_feat_z:
                    _, _, current_loss_x_soft, _ = ftrl_x_ensemble_soft.fit_soft_labels(feat_indices, buffer_feat_x[update_idx], combined_Softlabel_ensemble_soft, decay_choice,
                                                     contribute_error_rate)
                    _, _, current_loss_z_soft, _ = ftrl_z_ensemble_soft.fit_soft_labels(feat_indices, buffer_feat_z[update_idx], combined_Softlabel_ensemble_soft, decay_choice,
                                                     contribute_error_rate)

                    x_soft_loss += current_loss_x_soft
                    z_soft_loss += current_loss_z_soft
                    # 计算动态权重 Lambda
                    # 避免 exp 溢出的简单保护 (可选，此处保持公式原样)
                    val_x = -eta * x_soft_loss
                    val_z = -eta * z_soft_loss
                    max_val = np.maximum(val_x, val_z)
                    lamda_soft = np.exp(val_x - max_val) / (np.exp(val_x - max_val) + np.exp(val_z - max_val))
                    loss_x_history.append(current_loss_x_soft)
                    loss_z_history.append(current_loss_z_soft)



            temp_ftrl_x_ensemble_soft = copy.deepcopy(ftrl_x_ensemble_soft )
            temp_ftrl_z_ensemble_soft  = copy.deepcopy(ftrl_z_ensemble_soft )

            if soft_labels_ensemble_all_hard is not None:
                start_idx_pseudo = current_time - len(soft_labels_ensemble_all_hard)  # 通常是 update_idx + 1

                for k in range(len(soft_labels_ensemble_all_hard)):
                    t_idx = start_idx_pseudo + k

                    if t_idx > update_idx and t_idx in buffer_feat_x and t_idx in buffer_feat_z:

                        p_lbl_ens_soft = soft_labels_ensemble_all_soft[k]
                        if not np.isnan(p_lbl_ens_soft[0]):
                            _, _, temp_step_loss_x_soft, _ = temp_ftrl_x_ensemble_soft.fit_soft_labels(feat_indices, buffer_feat_x[t_idx],
                                                                                 p_lbl_ens_soft, decay_choice,
                                                                                 contribute_error_rate)
                            _, _, temp_step_loss_z_soft, _ = temp_ftrl_z_ensemble_soft.fit_soft_labels(feat_indices, buffer_feat_z[t_idx],
                                                                                 p_lbl_ens_soft, decay_choice,
                                                                                 contribute_error_rate)

            p_x_ensemble_soft, w_x_ensemble_soft = temp_ftrl_x_ensemble_soft.fit_soft_labels(feat_indices, feat_x_curr, 100, decay_choice, contribute_error_rate)
            soft_pred_label_x = 1 if p_x_ensemble_soft > 0.5 else 0
            p_z_ensemble_soft, w_z_ensemble_soft = temp_ftrl_z_ensemble_soft.fit_soft_labels(feat_indices, feat_z_curr, 100, decay_choice, contribute_error_rate)
            soft_pred_label_z = 1 if p_z_ensemble_soft > 0.5 else 0
            term_x_soft = np.dot(w_x_ensemble_soft, feat_x_curr)
            term_z_soft = np.dot(w_z_ensemble_soft, feat_z_curr)
            p_ens_soft = sigmoid(lamda_soft * term_x_soft + (1.0 - lamda_soft) * term_z_soft)
            pred_label_ens_soft = 1 if p_ens_soft > 0.5 else 0


            gama=0.5
            a,b=ftrl_x_ensemble_soft.fit_soft_labels(feat_indices, feat_x_curr, 100, decay_choice, contribute_error_rate)
            t_x=sigmoid(gama * np.dot(b, feat_x_curr) + (1-gama)* np.dot(w_x_ensemble_soft, feat_x_curr))
            resnet_soft_pred_label_x = 1 if t_x > 0.5 else 0
            a,b=ftrl_z_ensemble_soft.fit_soft_labels(feat_indices, feat_z_curr, 100, decay_choice, contribute_error_rate)
            t_z=sigmoid(gama * np.dot(b, feat_x_curr) + (1-gama)* np.dot(w_x_ensemble_soft, feat_x_curr))
            resnet_soft_pred_label_z = 1 if t_z > 0.5 else 0

            t_emb=0.5*t_z+0.5*t_x
            resnet_pred_label_ens_soft = 1 if t_emb > 0.5 else 0


            if update_idx in buffer_feat_x: del buffer_feat_x[update_idx]
            if update_idx in buffer_feat_z: del buffer_feat_z[update_idx]


            buffer_feat_x[current_time] = feat_x_curr
            buffer_feat_z[current_time] = feat_z_curr

            curr_true_y = Y_label[current_time]
            if current_time<DELAY_ROUNDS:
                continue

            results_lists['soft_X']['errors'].append(int(np.abs(curr_true_y - soft_pred_label_x) > 0.5))
            results_lists['soft_X']['preds'].append(soft_pred_label_x)
            results_lists['soft_X']['probs'].append(p_x_ensemble_soft)  # <--- 新增这行：保存 p_x

            results_lists['soft_Z']['errors'].append(int(np.abs(curr_true_y - soft_pred_label_z) > 0.5))
            results_lists['soft_Z']['preds'].append(soft_pred_label_z)
            results_lists['soft_Z']['probs'].append(p_z_ensemble_soft)  # <--- 新增这行：保存 p_z

            results_lists['soft_Ens']['errors'].append(int(np.abs(curr_true_y - pred_label_ens_soft) > 0.5))
            results_lists['soft_Ens']['preds'].append(pred_label_ens_soft)
            results_lists['soft_Ens']['probs'].append(p_ens_soft)


            results_lists['resnet_soft_X']['errors'].append(int(np.abs(curr_true_y - resnet_soft_pred_label_x) > 0.5))
            results_lists['resnet_soft_X']['preds'].append(resnet_soft_pred_label_x)
            results_lists['resnet_soft_X']['probs'].append(t_x)  # <--- 新增这行：保存 p_x

            results_lists['resnet_soft_Z']['errors'].append(int(np.abs(curr_true_y - resnet_soft_pred_label_z) > 0.5))
            results_lists['resnet_soft_Z']['preds'].append(resnet_soft_pred_label_z)
            results_lists['resnet_soft_Z']['probs'].append(t_z)  # <--- 新增这行：保存 p_z

            results_lists['resnet_soft_Ens']['errors'].append(int(np.abs(curr_true_y - resnet_pred_label_ens_soft) > 0.5))
            results_lists['resnet_soft_Ens']['preds'].append(resnet_pred_label_ens_soft)
            results_lists['resnet_soft_Ens']['probs'].append(t_emb)

        # 2. 准备数据字典
        # 将代码中的原始记录映射为评估器需要的格式
        eval_data = {
            'soft_X': results_lists['soft_X'],
            'soft_Z': results_lists['soft_Z'],
            'soft_Ens': results_lists['soft_Ens'],
            'resnet_soft_X': results_lists['resnet_soft_X'],
            'resnet_soft_Z': results_lists['resnet_soft_Z'],
            'resnet_soft_Ens': results_lists['resnet_soft_Ens'],
        }

        valid_len = len(results_lists['soft_X']['preds'])

        valid_labels = Y_label[DELAY_ROUNDS:valid_len+DELAY_ROUNDS]

        evaluator = DriftEvaluator(
            y_true=valid_labels,
            results_dict=eval_data,
            drift_start=DRIFT_START,
            delay_rounds=DELAY_ROUNDS,
            dataset=dataset,
            window_size=WINDOW_SIZE_ZONE
        )
        print(f"\n{'=' * 20} ADVANCED DRIFT METRICS {'=' * 20}")
        print(f"Drift Start: {DRIFT_START}, Delay: {DELAY_ROUNDS}")
        metrics_df = evaluator.get_metrics_table()
        print(metrics_df)
        print("=" * 60)

        print("Plotting curves...")
        evaluator.plot_curves(title=f"{dataset}: Impact of Label Delay ({DELAY_ROUNDS}) on Concept Drift Adaptation| BATCH_SIZE: {batch_size_denominator} | WINDOW_SIZE: {window_size_denominator}",BATCH_SIZE=batch_size_denominator,WINDOW_SIZE=window_size_denominator,save_dir="../Result/baseline_my_v1/"+dataset+"_"+str(decay_choice)+"_"+str(contribute_error_rate)+"_"+str(window_size_denominator)+"_"+str(batch_size_denominator) )

        def compute_final_stats(error_list, pred_list, prob_list, true_labels):
            cer_curve = np.cumsum(error_list) / (np.arange(len(error_list)) + 1.0)
            recall, precision, f1, auc = calculate_metrics(true_labels, pred_list, prob_list)
            return cer_curve, recall, precision, f1, auc

        soft_X_CER, soft_X_rec, soft_X_prec, soft_X_f1, soft_X_auc = compute_final_stats(
            results_lists['soft_X']['errors'],
            results_lists['soft_X']['preds'],
            results_lists['soft_X']['probs'],
            valid_labels
        )

        soft_Y_CER, soft_Y_rec, soft_Y_prec, soft_Y_f1, soft_Y_auc = compute_final_stats(
            results_lists['soft_Z']['errors'],
            results_lists['soft_Z']['preds'],
            results_lists['soft_Z']['probs'],
            valid_labels
        )

        soft_Z_CER, soft_Z_rec, soft_Z_prec, soft_Z_f1, soft_Z_auc = compute_final_stats(
            results_lists['soft_Ens']['errors'],
            results_lists['soft_Ens']['preds'],
            results_lists['soft_Ens']['probs'],
            valid_labels
        )

        resnet_soft_X_CER, resnet_soft_X_rec, resnet_soft_X_prec, resnet_soft_X_f1, resnet_soft_X_auc = compute_final_stats(
            results_lists['resnet_soft_X']['errors'],
            results_lists['resnet_soft_X']['preds'],
            results_lists['resnet_soft_X']['probs'],
            valid_labels
        )

        resnet_soft_Y_CER, resnet_soft_Y_rec, resnet_soft_Y_prec, resnet_soft_Y_f1, resnet_soft_Y_auc = compute_final_stats(
            results_lists['resnet_soft_Z']['errors'],
            results_lists['resnet_soft_Z']['preds'],
            results_lists['resnet_soft_Z']['probs'],
            valid_labels
        )

        resnet_soft_Z_CER, resnet_soft_Z_rec, resnet_soft_Z_prec, resnet_soft_Z_f1, resnet_soft_Z_auc = compute_final_stats(
            results_lists['resnet_soft_Ens']['errors'],
            results_lists['resnet_soft_Ens']['preds'],
            results_lists['resnet_soft_Ens']['probs'],
            valid_labels
        )


        temp = np.ones((n, 1))
        X_masked_df = pd.DataFrame(X_masked)
        X_zeros = X_masked_df.fillna(value=0).values
        X_input_zero = np.hstack((temp, X_zeros))

        svm_error, _ = calculate_svm_error(X_input_zero[DELAY_ROUNDS:], Y_label[DELAY_ROUNDS:n], len(X_input_zero[DELAY_ROUNDS:]))

        results = {}

        results['soft_X_imp_delay_missing'] = {
            'CER': soft_X_CER[-1], 'recall': soft_X_rec, 'precision': soft_X_prec, 'f1': soft_X_f1, 'auc': soft_X_auc
        }
        results['soft_Y_imp_delay_missing'] = {
            'CER': soft_Y_CER[-1], 'recall': soft_Y_rec, 'precision': soft_Y_prec, 'f1': soft_Y_f1, 'auc': soft_Y_auc
        }
        results['soft_Z_imp_delay_missing'] = {
            'CER': soft_Z_CER[-1], 'recall': soft_Z_rec, 'precision': soft_Z_prec, 'f1': soft_Z_f1, 'auc': soft_Z_auc
        }

        results['resnet_soft_X_imp_delay_missing'] = {
            'CER': resnet_soft_X_CER[-1], 'recall': resnet_soft_X_rec, 'precision': resnet_soft_X_prec, 'f1': resnet_soft_X_f1, 'auc': resnet_soft_X_auc
        }
        results['resnet_soft_Y_imp_delay_missing'] = {
            'CER': resnet_soft_Y_CER[-1], 'recall': resnet_soft_Y_rec, 'precision': resnet_soft_Y_prec, 'f1': resnet_soft_Y_f1, 'auc': resnet_soft_Y_auc
        }
        results['resnet_soft_Z_imp_delay_missing'] = {
            'CER': resnet_soft_Z_CER[-1], 'recall': resnet_soft_Z_rec, 'precision': resnet_soft_Z_prec, 'f1': resnet_soft_Z_f1, 'auc': resnet_soft_Z_auc
        }

        results['SVM_delay_missing'] = {
            'CER': svm_error, 'recall': 'N/A', 'precision': 'N/A', 'f1': 'N/A', 'auc': 'N/A'
        }
        print(f"Final Experiment Results: {dataset} | Delay: {DELAY_ROUNDS}")
        #results_df = print_results_table(results)
        print(pd.DataFrame(results).T)
        print("\n")

