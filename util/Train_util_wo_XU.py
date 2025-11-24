
import math
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler

def normalize_numpyarray(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    arr_normalized = (arr - arr_min) / (arr_max - arr_min)
    return arr_normalized

def normalize_dataframe(df_data):
    scaler =MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_data), columns=df_data.columns)
    return df_normalized

def load_data_pkl(pos_path, neg_path,p=1,n=0):
    with open(pos_path, "rb") as tf:
        feature_dict = pickle.load(tf)
    if isinstance(feature_dict, dict):
        pos = np.array(list(feature_dict.values()))
    else:
        pos = feature_dict  # 如果已经是 NumPy 数组，直接使用
    # pos = np.array([item for item in feature_dict.values()])
    pos = np.insert(pos, 0, values=[p for _ in range(pos.shape[0])], axis=1)
    print("pos:", pos.shape)
    # print("pos[0] type:", type(pos[0]), "content:", pos[0])
    with open(neg_path, "rb") as tf:
        feature_dict = pickle.load(tf)
    if isinstance(feature_dict, dict):
        neg = np.array(list(feature_dict.values()))
    else:
        neg = feature_dict  # 如果已经是 NumPy 数组，直接使用
    # neg = np.array([item for item in feature_dict.values()])
    neg = np.insert(neg, 0, values=[n for _ in range(neg.shape[0])], axis=1)
    print("neg:", neg.shape)
    # print("neg[0] type:", type(neg[0]), "content:", neg[0])
    data = np.row_stack((pos, neg))
    print("data", data.shape)
    data_Y, data_X = data[:, 0], data[:, 1:]
    print("label:", data_Y.shape)
    print("features:", data_X.shape)
    return data_Y, data_X


def load_data_csv(pos_path, neg_path):
    pos = pd.read_csv(pos_path)
    pos = pos.select_dtypes(include=[np.number])  # 只保留数值列
    pos = np.array(pos)
    pos = np.insert(pos, 0, values=[1 for _ in range(pos.shape[0])], axis=1)
    print("pos:", pos.shape)
    neg = pd.read_csv(neg_path)
    neg = neg.select_dtypes(include=[np.number])  # 只保留数值列
    neg = np.array(neg)
    neg = np.insert(neg, 0, values=[0 for _ in range(neg.shape[0])], axis=1)
    print("neg:", neg.shape)
    data = np.row_stack((pos, neg))
    print("data:", data.shape)
    data_Y, data_X = data[:, 0], data[:, 1:]
    print("label:", data_Y.shape)  
    print("features:", data_X.shape)  
    return data_Y, data_X

def get_CM(y, y_pre):
    cm = confusion_matrix(y, y_pre)
    # print(f"Confusion matrix:\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    # print(f"tp: {tp}, fn: {fn}")
    acc = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    pr = tp / (tp + fp)
    if sn + pr == 0:
        f1 = 0  # 或者用 np.nan 来表示无效结果
    else:
        f1 = (2 * sn * pr) / (sn + pr)

    return acc, sn, sp, mcc, pr, f1

def save_ROC_PR(data_y,data_X,model,path1,tag,teortr):
    y_pred_proba = model.predict_proba(data_X)[:, 1]
    auc = roc_auc_score(data_y, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(data_y,y_pred_proba)

    roc_curve_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
    roc_curve_data.to_csv(path1+'csv/'+tag+teortr+'_roc.csv', index=False)
 
    plt.plot(fpr, tpr, color='darkred', label='ROC (AUC = %0.3f)' % auc)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=15)
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(path1, tag + teortr+'_ROC.png'))
    ap = average_precision_score(data_y,y_pred_proba)
    precision, recall, thresholds = precision_recall_curve(data_y,y_pred_proba)

    pr_curve_data = pd.DataFrame({'Precision': precision, 'Recall': recall})
    pr_curve_data.to_csv(path1 + 'csv/' + tag + teortr + '_pr.csv', index=False)

    plt.plot(recall, precision, color='darkred', label='PR Vote\ (AP = %0.3f)' % ap)
    plt.plot([1, 0], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('P-R Curve', fontsize=15)
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(path1, tag +teortr+ 'PR.png'))

