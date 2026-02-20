import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def get_tra_zero_datastream(dataset):
    X = pd.read_csv("../dataset/MaskData/" + dataset + "/X_process.txt", sep=" ", header=None)
    X = X.values
    n = X.shape[0]
    feat = X.shape[1]
    perm = np.arange(n)
    np.random.seed(1)
    np.random.shuffle(perm)
    perm = np.array(perm)
    X = X[perm]
    X = X.tolist()
    star_row = 0
    end_column = 0
    X_trapezoid = []
    X_masked = []
    X_zeros = []
    for i in range(5):
        end_row = star_row + math.ceil(n / 5)
        if end_row > n: end_row = n
        end_column = end_column + math.ceil(feat / 5)
        if end_column > feat: end_column = feat
        for j in range(star_row, end_row):
            row_1 = X[j][0:end_column]
            row_2 = row_1 + [np.nan] * (feat - end_column)
            row_3 = row_1 + [0] * (feat - end_column)

            X_trapezoid.append(row_1)
            X_masked.append(row_2)
            X_zeros.append(row_3)
        star_row = end_row

    path = "../dataset/MaskData/" + dataset
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path + "/X.txt", np.array(X_masked))
    np.savetxt(path + "/X_trapezoid_zeros.txt", np.array(X_zeros))
    file = open(path + "/X_trapezoid.txt", 'w')
    for fp in X_trapezoid:
        file.write(str(fp))
        file.write('\n')
    file.close()

def chack_Nan(X_masked,n):
    for i in range(n):
        X_masked[i] = X_masked[i].strip()
        X_masked[i] = X_masked[i].strip("[]")
        X_masked[i] = X_masked[i].split(",")
        X_masked[i] = list(map(float, X_masked[i]))
        narry = np.array(X_masked[i])
        where_are_nan = np.isnan(narry)
        narry[where_are_nan] = 0
        X_masked[i] = narry.tolist()
    return X_masked

def get_cont_indices(X):
    max_ord=14
    indices = np.zeros(X.shape[1]).astype(bool)
    for i, col in enumerate(X.T):
        col_nonan = col[~np.isnan(col)]
        col_unique = np.unique(col_nonan)
        if len(col_unique) > max_ord:
            indices[i] = True
    return indices

def cont_to_binary(x):
    while True:
        cutoff = np.random.choice(x)
        if len(x[x < cutoff]) > 0.1*len(x) and len(x[x < cutoff]) < 0.9*len(x):
            break
    return (x > cutoff).astype(int)

def cont_to_ord(x, k):
    std_dev = np.std(x)
    cuttoffs = np.linspace(np.min(x), np.max(x), k+1)[1:]
    ords = np.zeros(len(x))
    for cuttoff in cuttoffs:
        ords += (x > cuttoff).astype(int)
    return ords.astype(int)

def get_mae(x_imp, x_true, x_obs=None):
    if x_obs is not None:
        loc = np.isnan(x_obs)
        imp = x_imp[loc]
        val = x_true[loc]
        return np.mean(np.abs(imp - val))
    else:
        return np.mean(np.abs(x_imp - x_true))

def get_smae(x_imp, x_true, x_obs, Med=None, per_type=False, cont_loc=None, bin_loc=None, ord_loc=None):
    error = np.zeros((x_obs.shape[1],2))
    for i, col in enumerate(x_obs.T):
        test = np.bitwise_and(~np.isnan(x_true[:,i]), np.isnan(col))
        if np.sum(test) == 0:
            error[i,0] = np.nan
            error[i,1] = np.nan
            continue
        col_nonan = col[~np.isnan(col)]
        x_true_col = x_true[test,i]
        x_imp_col = x_imp[test,i]
        if Med is not None:
            median = Med[i]
        else:
            median = np.median(col_nonan)
        diff = np.abs(x_imp_col - x_true_col)
        med_diff = np.abs(median - x_true_col)
        error[i,0] = np.sum(diff)
        error[i,1]= np.sum(med_diff)
    if per_type:
        if not cont_loc:
            cont_loc = [True] * 5 + [False] * 10
        if not bin_loc:
            bin_loc = [False] * 5 + [True] * 5 + [False] * 5
        if not ord_loc:
            ord_loc = [False] * 10 + [True] * 5
        loc = [cont_loc, bin_loc, ord_loc]
        scaled_diffs = np.zeros(3)
        for j in range(3):
            scaled_diffs[j] = np.sum(error[loc[j],0])/np.sum(error[loc[j],1])
    else:
        scaled_diffs = error[:,0] / error[:,1]
    return scaled_diffs

def get_smae_per_type(x_imp, x_true, x_obs, cont_loc=None, bin_loc=None, ord_loc=None):
    if not cont_loc:
        cont_loc = [True] * 5 + [False] * 10
    if not bin_loc:
        bin_loc = [False] * 5 + [True] * 5 + [False] * 5
    if not ord_loc:
        ord_loc = [False] * 10 + [True] * 5
    loc = [cont_loc, bin_loc, ord_loc]
    scaled_diffs = np.zeros(3)
    for j in range(3):
        missing = np.isnan(x_obs[:,loc[j]])
        med = np.median(x_obs[:,loc[j]][~missing])
        diff = np.abs(x_imp[:,loc[j]][missing] - x_true[:,loc[j]][missing])
        med_diff = np.abs(med - x_true[:,loc[j]][missing])
        scaled_diffs[j] = np.sum(diff)/np.sum(med_diff)
    return scaled_diffs

def get_rmse(x_imp, x_true, relative=False):
    diff = x_imp - x_true
    mse = np.mean(diff**2.0, axis=0)
    rmse = np.sqrt(mse)
    return rmse if not relative else rmse/np.sqrt(np.mean(x_true**2))

def get_relative_rmse(x_imp, x_true, x_obs):
    loc = np.isnan(x_obs)
    imp = x_imp[loc]
    val = x_true[loc]
    return get_scaled_error(imp, val)

def get_scaled_error(sigma_imp, sigma):
    return np.linalg.norm(sigma - sigma_imp) / np.linalg.norm(sigma)

def mask_types(X, mask_num, seed):
    X_masked = np.copy(X).astype(float)
    mask_indices = []
    num_rows = X_masked.shape[0]
    num_cols = X_masked.shape[1]
    for i in range(num_rows):
        np.random.seed(seed*num_rows-i)
        for j in range(num_cols//2):
            rand_idx=np.random.choice(2,mask_num,False)
            for idx in rand_idx:
                X_masked[i,idx+2*j]=np.nan
                mask_indices.append((i, idx+2*j))
    return X_masked

def mask(X, mask_fraction, seed=0, verbose=False):
    complete = False
    count = 0
    X_masked = np.copy(X)
    obs_indices = np.argwhere(~np.isnan(X))
    total_observed = len(obs_indices)
    while not complete:
        np.random.seed(seed)
        if (verbose): print(seed)
        mask_indices = obs_indices[np.random.choice(len(obs_indices), size=int(mask_fraction*total_observed), replace=False)]
        for i,j in mask_indices:
            X_masked[i,j] = np.nan
        complete = True
        for row in X_masked:
            if len(row[~np.isnan(row)]) == 0:
                seed += 1
                count += 1
                complete = False
                X_masked = np.copy(X)
                break
        if count == 50:
            raise ValueError("Failure in Masking data without empty rows")
    return X_masked, mask_indices, seed

def mask_per_row(X, seed=0, size=1):
    X_masked = np.copy(X)
    n,p = X.shape
    for i in range(n):
        np.random.seed(seed*n+i)
        rand_idx = np.random.choice(p, size)
        X_masked[i,rand_idx] = np.nan
    return X_masked

def _project_to_correlation(covariance):
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def generate_sigma(seed):
    np.random.seed(seed)
    W = np.random.normal(size=(18, 18))
    covariance = np.matmul(W, W.T)
    D = np.diagonal(covariance)
    D_neg_half = np.diag(1.0/np.sqrt(D))
    return np.matmul(np.matmul(D_neg_half, covariance), D_neg_half)

def continuous2ordinal(x, k = 2, cutoff = None):
    q = np.quantile(x, (0.05,0.95))
    if k == 2:
        if cutoff is None:
            cutoff = np.random.choice(x[(x > q[0])*(x < q[1])])
        x = (x >= cutoff).astype(int)
    else:
        if cutoff is None:
            std_dev = np.std(x)
            min_cutoff = np.min(x) - 0.1 * std_dev
            cutoff = np.sort(np.random.choice(x[(x > q[0])*(x < q[1])], k-1, False))
            max_cutoff = np.max(x) + 0.1 * std_dev
            cuttoff = np.hstack((min_cutoff, cutoff, max_cutoff))
        x = np.digitize(x, cuttoff)
    return x

def grassman_dist(A,B):
    U1, d1, _ = np.linalg.svd(A, full_matrices = False)
    U2, d2, _ = np.linalg.svd(B, full_matrices = False)
    _, d,_    = np.linalg.svd(np.dot(U1.T, U2))
    theta     = np.arccos(d)
    return np.linalg.norm(theta), np.linalg.norm(d1-d2)

def get_cap_hyperparameter(dataset):
    decay_choices            = {"ionosphere": 4,     "wbc": 2,    "wdbc": 2,     "german": 4,     "diabetes": 4,     "credit": 4,         "australian":3,      "wpbc": 4,     "kr_vs_kp":4,     "svmguide3":4,    "magic04":4,    "imdb":2,     "synthetic":4     ,"a8a": 4     , "splice":4    , "dna": 4     , "hapt": 4     , "Stream1": 4,"spambase":4,"AGRa":4,"AGRg":4,"kr-vs-kp":4,"HYPa":4,"RBF20_AbruptGradual_10k":4,"RBF20_FastGradual_10k":4,"RBF20_FastGradual_10k_1000draft_10n":4,"RBF20_FastGradual_10k_2000draft_10n":4,"RBF20_FastGradual_10k_2000draft_20n":4,"RBF20_FastGradual_10k_3000draft_20n":4,"RBF20_AbruptGradual_10k_1draft_20a_10n":4,"RBF20_AbruptGradual_10k_1draft_20a_20n":4,"RBF20_AbruptGradual_10k_1draft_100a_20n":4,"RBF20_AbruptGradual_10k_1draft_100a_10n":4,"RBF20_FastGradual_10k_1000draft_20n":4,"RBF20_FastGradual_20k_1000draft_20n":4,"RBF20_FastGradual_20k_2000draft_20n":4,"RBF20_FastGradual_20k_3000draft_20n":4,"RBF20_FastGradual_20k_4000draft_20n":4,"R_LUdata":4,"R_chessweka":4,"R_phishing":4,"R_spam":4,"R_weather":4,"Agrawal9_AbruptGradual_10k":4,"Hyperplane20_AbruptGradual_10k":4,"SEA3_AbruptGradual_10k":4,"Agrawal9_AbruptGradual_50k":4,"Hyperplane20_AbruptGradual_50k":4,"SEA3_AbruptGradual_50k":4,"Agrawal9_FastGradual_10k_1000draft":4,"Agrawal9_FastGradual_10k_2000draft":4,"Hyperplane20_FastGradual_10k_1000draft":4,"Hyperplane20_FastGradual_10k_2000draft":4,"SEA3_FastGradual_10k_1000draft":4,"SEA3_FastGradual_10k_2000draft":4,"Agrawal9_FastGradual_10k_3000draft":4,"Hyperplane20_FastGradual_10k_3000draft":4,"SEA3_FastGradual_10k_3000draft":4}
    contribute_error_rates   = {"ionosphere": 0.02,  "wbc": 0.05, "wdbc": 0.05,  "german": 0.01, "diabetes": 0.02,  "credit": 0.01,      "australian":0.02,   "wpbc": 0.01,  "kr_vs_kp":0.01,  "svmguide3":0.01, "magic04":0.01, "imdb":0.01,  "synthetic":0.01  ,"a8a": 0.01  , "splice":0.01 , "dna": 0.01  , "hapt": 0.01  , "Stream1": 0.01,"spambase":0.01,"AGRa":0.01,"AGRg":0.01,"kr-vs-kp":0.01,"HYPa":0.01,"RBF20_AbruptGradual_10k":0.01,"RBF20_FastGradual_10k":0.01,"RBF20_FastGradual_10k_1000draft_10n":0.01,"RBF20_FastGradual_10k_2000draft_10n":0.01,"RBF20_FastGradual_10k_2000draft_20n":0.01,"RBF20_FastGradual_10k_3000draft_20n":0.01,"RBF20_AbruptGradual_10k_1draft_20a_10n":0.01,"RBF20_AbruptGradual_10k_1draft_20a_20n":0.01,"RBF20_AbruptGradual_10k_1draft_100a_20n":0.01,"RBF20_AbruptGradual_10k_1draft_100a_10n":0.01,"RBF20_FastGradual_10k_1000draft_20n":0.01,"RBF20_FastGradual_20k_1000draft_20n":0.01,"RBF20_FastGradual_20k_2000draft_20n":0.01,"RBF20_FastGradual_20k_3000draft_20n":0.05,"RBF20_FastGradual_20k_4000draft_20n":0.005,"R_LUdata":0.01,"R_chessweka":0.005,"R_phishing":0.01,"R_spam":0.01,"R_weather":0.01,"Agrawal9_AbruptGradual_10k":0.01,"Hyperplane20_AbruptGradual_10k":0.01,"SEA3_AbruptGradual_10k":0.01,"Agrawal9_AbruptGradual_50k":0.0005,"Hyperplane20_AbruptGradual_50k":0.05,"SEA3_AbruptGradual_50k":0.01,"Agrawal9_FastGradual_10k_1000draft":0.01,"Agrawal9_FastGradual_10k_2000draft":0.01,"Hyperplane20_FastGradual_10k_1000draft":0.01,"Hyperplane20_FastGradual_10k_2000draft":0.05,"SEA3_FastGradual_10k_1000draft":0.01,"SEA3_FastGradual_10k_2000draft":0.01,"Agrawal9_FastGradual_10k_3000draft":0.01,"Hyperplane20_FastGradual_10k_3000draft":0.05,"SEA3_FastGradual_10k_3000draft":0.01}
    window_size_denominators = {"ionosphere": 2,     "wbc": 8,    "wdbc": 6,     "german": 8,     "diabetes": 2,     "credit": 4,         "australian":4   ,   "wpbc": 8,     "kr_vs_kp":2,     "svmguide3":2,    "magic04":2,    "imdb":2,     "synthetic":2     ,"a8a": 2     , "splice":4    , "dna": 4     , "hapt": 4     , "Stream1": 4,"spambase":4,"AGRa":4,"AGRg":4,"kr-vs-kp":4,"HYPa":4,"RBF20_AbruptGradual_10k":4,"RBF20_FastGradual_10k":4,"RBF20_FastGradual_10k_1000draft_10n":4,"RBF20_FastGradual_10k_2000draft_10n":4,"RBF20_FastGradual_10k_2000draft_20n":4,"RBF20_FastGradual_10k_3000draft_20n":4,"RBF20_AbruptGradual_10k_1draft_20a_10n":4,"RBF20_AbruptGradual_10k_1draft_20a_20n":4,"RBF20_AbruptGradual_10k_1draft_100a_20n":4,"RBF20_AbruptGradual_10k_1draft_100a_10n":4,"RBF20_FastGradual_10k_1000draft_20n":4,"RBF20_FastGradual_20k_1000draft_20n":8,"RBF20_FastGradual_20k_2000draft_20n":8,"RBF20_FastGradual_20k_3000draft_20n":8,"RBF20_FastGradual_20k_4000draft_20n":8,"R_LUdata":4,"R_chessweka":2,"R_phishing":4,"R_spam":4,"R_weather":4,"Agrawal9_AbruptGradual_10k":4,"Hyperplane20_AbruptGradual_10k":4,"SEA3_AbruptGradual_10k":4,"Agrawal9_AbruptGradual_50k":20,"Hyperplane20_AbruptGradual_50k":20,"SEA3_AbruptGradual_50k":20,"Agrawal9_FastGradual_10k_1000draft":4,"Agrawal9_FastGradual_10k_2000draft":4,"Hyperplane20_FastGradual_10k_1000draft":4,"Hyperplane20_FastGradual_10k_2000draft":4,"SEA3_FastGradual_10k_1000draft":4,"SEA3_FastGradual_10k_2000draft":4,"Agrawal9_FastGradual_10k_3000draft":4,"Hyperplane20_FastGradual_10k_3000draft":4,"SEA3_FastGradual_10k_3000draft":4}
    batch_size_denominators  = {"ionosphere": 5,    "wbc": 10,    "wdbc": 8,     "german": 12,     "diabetes": 9,     "credit": 4,         "australian": 8 ,  "wpbc": 8,     "kr_vs_kp":8,     "svmguide3":8,    "magic04":16,    "imdb":8,     "synthetic":8    ,"a8a": 8     , "splice":4    , "dna": 8     , "hapt": 8     , "Stream1": 8,"spambase":8,"AGRa":8,"AGRg":8,"kr-vs-kp":40,"HYPa":16,"RBF20_AbruptGradual_10k":8,"RBF20_FastGradual_10k":8,"RBF20_FastGradual_10k_1000draft_10n":8,"RBF20_FastGradual_10k_2000draft_10n":8,"RBF20_FastGradual_10k_2000draft_20n":8,"RBF20_FastGradual_10k_3000draft_20n":8,"RBF20_AbruptGradual_10k_1draft_20a_10n":8,"RBF20_AbruptGradual_10k_1draft_20a_20n":8,"RBF20_AbruptGradual_10k_1draft_100a_20n":8,"RBF20_AbruptGradual_10k_1draft_100a_10n":8,"RBF20_FastGradual_10k_1000draft_20n":8,"RBF20_FastGradual_20k_1000draft_20n":16,"RBF20_FastGradual_20k_2000draft_20n":16,"RBF20_FastGradual_20k_3000draft_20n":16,"RBF20_FastGradual_20k_4000draft_20n":16,"R_LUdata":24,"R_chessweka":6,"R_phishing":8,"R_spam":8,"R_weather":8,"Agrawal9_AbruptGradual_10k":8,"Hyperplane20_AbruptGradual_10k":8,"SEA3_AbruptGradual_10k":8,"Agrawal9_AbruptGradual_50k":40,"Hyperplane20_AbruptGradual_50k":40,"SEA3_AbruptGradual_50k":40,"Agrawal9_FastGradual_10k_1000draft":8,"Agrawal9_FastGradual_10k_2000draft":8,"Hyperplane20_FastGradual_10k_1000draft":8,"Hyperplane20_FastGradual_10k_2000draft":8,"SEA3_FastGradual_10k_1000draft":8,"SEA3_FastGradual_10k_2000draft":8,"Agrawal9_FastGradual_10k_3000draft":8,"Hyperplane20_FastGradual_10k_3000draft":8,"SEA3_FastGradual_10k_3000draft":8}
    shuffles                 = {"ionosphere": True,  "wbc": False,"wdbc": True,  "german": False, "diabetes": True,  "credit": True,      "australian":False , "wpbc": False, "kr_vs_kp":True, "svmguide3":True, "magic04":True, "imdb":False, "synthetic":False ,"a8a": False , "splice":True , "dna": False , "hapt": False , "Stream1":False,"spambase":True,"AGRa":False,"AGRg":False,"kr-vs-kp":True,"HYPa":False,"RBF20_AbruptGradual_10k":False,"RBF20_FastGradual_10k":False,"RBF20_FastGradual_10k_1000draft_10n":False,"RBF20_FastGradual_10k_2000draft_10n":False,"RBF20_FastGradual_10k_2000draft_20n":False,"RBF20_FastGradual_10k_3000draft_20n":False,"RBF20_AbruptGradual_10k_1draft_20a_10n":False,"RBF20_AbruptGradual_10k_1draft_20a_20n":False,"RBF20_AbruptGradual_10k_1draft_100a_20n":False,"RBF20_AbruptGradual_10k_1draft_100a_10n":False,"RBF20_FastGradual_10k_1000draft_20n":False,"RBF20_FastGradual_20k_1000draft_20n":False,"RBF20_FastGradual_20k_2000draft_20n":False,"RBF20_FastGradual_20k_3000draft_20n":False,"RBF20_FastGradual_20k_4000draft_20n":False,"R_LUdata":False,"R_chessweka":False,"R_phishing":False,"R_spam":False,"R_weather":False,"Agrawal9_AbruptGradual_10k":False,"Hyperplane20_AbruptGradual_10k":False,"SEA3_AbruptGradual_10k":False,"Agrawal9_AbruptGradual_50k":False,"Hyperplane20_AbruptGradual_50k":False,"SEA3_AbruptGradual_50k":False,"Agrawal9_FastGradual_10k_1000draft":False,"Agrawal9_FastGradual_10k_2000draft":False,"Hyperplane20_FastGradual_10k_1000draft":False,"Hyperplane20_FastGradual_10k_2000draft":False,"SEA3_FastGradual_10k_1000draft":False,"SEA3_FastGradual_10k_2000draft":False,"Agrawal9_FastGradual_10k_3000draft":False,"Hyperplane20_FastGradual_10k_3000draft":False,"SEA3_FastGradual_10k_3000draft":False}

    batch_size_denominator=batch_size_denominators[dataset]
    decay_coef_change=0
    contribute_error_rate=contribute_error_rates[dataset]
    window_size_denominator=window_size_denominators[dataset]
    shuffle=shuffles[dataset]
    decay_choice=decay_choices[dataset]

    return contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,shuffle

def get_my_cap_hyperparameter(dataset):
    decay_choices            = {"Agrawal_9_5k_abrupt_2500drift":4,"SEA_3_5k_abrupt_2500drift":4,"Hyperplane_10_5k_abrupt_2500drift":4,"RandomRBF_10_5k_abrupt_2500drift":4,"SEA_3_5k_2abrupt_1500_2500":4,"SEA_3_5k_3abrupt_1500_2500_3500":4,"Hyperplane_10_5k_2abrupt_1500_2500":4,"Hyperplane_10_5k_3abrupt_1500_2500_3500":4}
    contribute_error_rates   = {"Agrawal_9_5k_abrupt_2500drift":0.01,"SEA_3_5k_abrupt_2500drift":0.01,"Hyperplane_10_5k_abrupt_2500drift":0.005,"RandomRBF_10_5k_abrupt_2500drift":0.01,"SEA_3_5k_2abrupt_1500_2500":0.01,"SEA_3_5k_3abrupt_1500_2500_3500":0.01,"Hyperplane_10_5k_2abrupt_1500_2500":0.01,"Hyperplane_10_5k_3abrupt_1500_2500_3500":0.01}
    window_size_denominators = {"Agrawal_9_5k_abrupt_2500drift":4,"SEA_3_5k_abrupt_2500drift":4,"Hyperplane_10_5k_abrupt_2500drift":10,"RandomRBF_10_5k_abrupt_2500drift":10,"SEA_3_5k_2abrupt_1500_2500":4,"SEA_3_5k_3abrupt_1500_2500_3500":4,"Hyperplane_10_5k_2abrupt_1500_2500":10,"Hyperplane_10_5k_3abrupt_1500_2500_3500":10}
    batch_size_denominators  = {"Agrawal_9_5k_abrupt_2500drift":10,"SEA_3_5k_abrupt_2500drift":10,"Hyperplane_10_5k_abrupt_2500drift":20,"RandomRBF_10_5k_abrupt_2500drift":20,"SEA_3_5k_2abrupt_1500_2500":10,"SEA_3_5k_3abrupt_1500_2500_3500":10,"Hyperplane_10_5k_2abrupt_1500_2500":20,"Hyperplane_10_5k_3abrupt_1500_2500_3500":20}
    shuffles                 = {"Agrawal_9_5k_abrupt_2500drift":False,"SEA_3_5k_abrupt_2500drift":False,"Hyperplane_10_5k_abrupt_2500drift":False,"RandomRBF_10_5k_abrupt_2500drift":False,"SEA_3_5k_2abrupt_1500_2500":False,"SEA_3_5k_3abrupt_1500_2500_3500":False,"Hyperplane_10_5k_2abrupt_1500_2500":False,"Hyperplane_10_5k_3abrupt_1500_2500_3500":False}
    batch_size_denominator=batch_size_denominators[dataset]
    decay_coef_change=0
    contribute_error_rate=contribute_error_rates[dataset]
    window_size_denominator=window_size_denominators[dataset]
    shuffle=shuffles[dataset]
    decay_choice=decay_choices[dataset]

    return contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,shuffle

def Cumulative_error_rate_semi_ensemble(predict_label_train_x_ensemble, predict_label_train_z_ensemble, Y_label):
    n = len(predict_label_train_x_ensemble)
    errors_x = []
    errors_z = []

    for i in range(n):
        y = Y_label[i]
        x = predict_label_train_x_ensemble[i]
        z = predict_label_train_z_ensemble[i]
        error_x = [int(np.abs(y - x) > 0.5)]
        error_z = [int(np.abs(y - z) > 0.5)]
        errors_x.append(error_x)
        errors_z.append(error_z)

    X_CER_ensemble = np.cumsum(errors_x) / (np.arange(len(errors_x)) + 1.0)
    Z_CER_ensemble = np.cumsum(errors_z) / (np.arange(len(errors_z)) + 1.0)

    l = len(X_CER_ensemble)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER of X & Z")
    x = range(l)

    X_CER_ensemble_line, = plt.plot(x, X_CER_ensemble, color='green', linestyle="--")
    Z_CER_ensemble_line, = plt.plot(x, Z_CER_ensemble, color='blue',  linestyle="-")

    plt.legend(handles=[X_CER_ensemble_line, Z_CER_ensemble_line], labels=["X_CER_ensemble_line", "Z_CER_ensemble_line"])

    plt.title("The Cumulative error rate(CER) of X_CER_ensemble_line, Z_CER_ensemble_line")
    plt.show()

def Cumulative_error_rate_semi(predict_label_train_x_ensemble, predict_label_train_z_ensemble, Y_label_fill_x,
                                Y_label_fill_z, Y_label, dataset):
    n = len(predict_label_train_x_ensemble)
    errors_x = []
    errors_z = []
    errors_x_ensemble = []
    errors_z_ensemble = []
    dataset = dataset
    for i in range(n):
        y = Y_label[i]
        x = Y_label_fill_x[i]
        z = Y_label_fill_z[i]
        x_ensemble = predict_label_train_x_ensemble[i]
        z_ensemble = predict_label_train_z_ensemble[i]

        error_x = [int(np.abs(y - x) > 0.5)]
        error_z = [int(np.abs(y - z) > 0.5)]
        error_x_ensemble = [int(np.abs(y - x_ensemble) > 0.5)]
        error_z_ensemble = [int(np.abs(y - z_ensemble) > 0.5)]

        errors_x.append(error_x)
        errors_z.append(error_z)
        errors_x_ensemble.append(error_x_ensemble)
        errors_z_ensemble.append(error_z_ensemble)

    X_CER = np.cumsum(errors_x) / (np.arange(len(errors_x)) + 1.0)
    Z_CER = np.cumsum(errors_z) / (np.arange(len(errors_z)) + 1.0)
    X_CER_ensemble = np.cumsum(errors_x_ensemble) / (np.arange(len(errors_x_ensemble)) + 1.0)
    Z_CER_ensemble = np.cumsum(errors_z_ensemble) / (np.arange(len(errors_z_ensemble)) + 1.0)

    l = len(X_CER_ensemble)
    plt.figure(figsize=(16, 10))
    plt.ylim((0, 1))
    plt.xlim((0, n))
    plt.ylabel("CER of X & Z")
    x = range(l)

    X_CER_line, = plt.plot(x, X_CER, color='pink', linestyle="-.")
    Z_CER_line, = plt.plot(x, Z_CER, color='black', linestyle=":")

    X_CER_ensemble_line, = plt.plot(x, X_CER_ensemble, color='green', linestyle="--")
    Z_CER_ensemble_line, = plt.plot(x, Z_CER_ensemble, color='blue',  linestyle="-")

    plt.legend(handles=[X_CER_ensemble_line, Z_CER_ensemble_line, X_CER_line, Z_CER_line],
                labels=["X_CER_ensemble_line", "Z_CER_ensemble_line", "X_CER_line", "Z_CER_line"])

    plt.title(dataset + "The Cumulative error rate(CER) of X_CER_ensemble_line, Z_CER_ensemble_line, X_CER_line, Z_CER_line")
    plt.show()

def shuffle_dataset_1(data):
    da1index = data[data.values == 1]
    da1index = da1index.index
    da1index = da1index.tolist()

    da0index = data[data.values == 0]
    da0index = da0index.index
    da0index = da0index.tolist()

    shufflearray = []
    j = 0
    for i in range(len(da1index)):
        shufflearray.append(da1index[i])
        if i % 3 == 0:
            shufflearray.append(da0index[j])
            j += 1

    for n in range(j, len(da0index)):
        shufflearray.append(da0index[n])

    shufflearray = np.array(shufflearray)
    shufflearray.flatten()
    return shufflearray

def Con2Ord(data, low, hight, Ord_num):
    needTopro = data.iloc[:, low : hight]
    for i in range(low, hight):
        data[i] = pd.cut(x=needTopro.loc[:, i], bins=Ord_num, labels=range(0, Ord_num))
    return data

def Con2Bin(data, low, hight):
    needTopro = data.iloc[:, low: hight]
    for i in range(low, hight):
        data[i] = pd.cut(x = needTopro.loc[:, i], bins=2, labels= range(0,2))
    return data

def calculate_discreteness(data, col_idx):
    col_data = data.iloc[:, col_idx]
    unique_ratio = len(col_data.unique()) / len(col_data)
    return unique_ratio

def auto_process_columns(data, threshold_low=0.1, threshold_high=0.5):
    n_columns = data.shape[1]

    discreteness_scores = []
    for i in range(n_columns):
        score = calculate_discreteness(data, i)
        discreteness_scores.append((i, score))

    sorted_scores = sorted(discreteness_scores, key=lambda x: x[1])

    low_idx = int(len(sorted_scores) * 1 / 3)
    high_idx = int(len(sorted_scores) * 2 / 3)

    low_discrete = [idx for idx, _ in sorted_scores[:low_idx]]
    medium_discrete = [idx for idx, _ in sorted_scores[low_idx:high_idx]]
    high_discrete = [idx for idx, _ in sorted_scores[high_idx:]]

    for i in range(n_columns):
        if i in low_discrete:
            data[i] = pd.cut(x=data.iloc[:, i], bins=2, labels=range(0, 2))
        elif i in medium_discrete:
            data[i] = pd.cut(x=data.iloc[:, i], bins=5, labels=range(0, 5))

    return data

def process_by_discreteness(data, method='auto', threshold_low=0.1, threshold_high=0.5):
    if method == 'auto':
        return auto_process_columns(data, threshold_low, threshold_high)
    else:
        data_row = data.shape[1] // 3
        low, mid, hight, final = 0, (data_row * 1), (data_row * 2), data.shape[1]
        data = Con2Bin(data, mid, hight)
        data = Con2Ord(data, hight, final, Ord_num=5)
        return data

def check_column_types_simple(X_target):
    is_continuous = []

    for col in X_target.columns:
        has_decimal = (X_target[col] % 1 != 0).any()

        if has_decimal:
            is_continuous.append(True)
        else:
            unique_count = X_target[col].nunique()
            is_continuous.append(unique_count >= 14)

    return any(is_continuous) and (not all(is_continuous))