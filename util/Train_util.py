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
    denominator = math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0
    # mcc = (tp * tn - fp * fn) / math.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0
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

def load_data_pkl(pos_path, neg_path,p=1,n=0):
    # with open(pos_path, "rb") as tf:
    #     feature_dict = pickle.load(tf)
    import joblib
    feature_dict = joblib.load(pos_path)

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

    data = np.vstack((pos, neg))
    print(f"Combined data shape: {data.shape}")

    data_Y = data[:, 0].astype(int)
    data_X = data[:, 1:] # Features

    print(f"Initial split: Labels shape {data_Y.shape}, Features shape {data_X.shape}, Features dtype {data_X.dtype}")

    # --- Check for Nested Features ---
    original_shape = data_X.shape
    processed = False # Flag to track if processing occurred

    if data_X.ndim == 2 and data_X.shape[1] == 2:
        processed = True # Mark that we entered this block
        print(f"Detected features shape {data_X.shape}. Checking if second column contains nested arrays.")
        is_nested = False
        if data_X.shape[0] > 0:
             first_element_col1 = data_X[0, 1]
             if isinstance(first_element_col1, (list, np.ndarray)):
                 is_nested = True
                 print("Second column seems to contain list or ndarray. Attempting to stack.")
             else:
                 print(f"Second column contains type {type(first_element_col1)}. Assuming these are the actual two features.")

        if is_nested:
            try:
                nested_features = data_X[:, 1]
                stacked_features = np.stack(nested_features)
                print(f"Successfully stacked nested features. Original shape: {original_shape}, New features shape: {stacked_features.shape}")
                data_X = stacked_features # Use stacked features
            except Exception as e:
                print(f"Error stacking features from the second column: {e}")
                print(f"Warning: Stacking failed. Returning ONLY the second column ({data_X.shape[0]}, ...) containing nested objects.")
                # *** CHANGE HERE: Return only the second column if stacking fails ***
                data_X = data_X[:, 1]
                print("Note: Further processing (e.g., padding/truncation) will be needed for these nested objects.")
        else:
             # Shape is (N, 2) but second column isn't list/array
             print("Shape is (N, 2) but second column not detected as nested.")
             # *** CHANGE HERE: Check if first column is numeric. If not, return only second column ***
             try:
                 # Try converting first column to float. If it fails, it contains strings.
                 _ = data_X[:, 0].astype(float)
                 print("First column appears numeric. Returning both columns as features.")
                 # Try converting both to float now
                 try:
                     data_X = data_X.astype(float)
                 except ValueError:
                     print("Warning: Could not convert both columns to float. Check data types.")
                     # Decide: return original or just second col? Let's return original with warning.
                     pass # Keep data_X as is, let downstream handle it maybe? Or return data_X[:, 1]?
             except ValueError:
                 print("Warning: First column contains non-numeric data (like strings). Returning ONLY the second column.")
                 data_X = data_X[:, 1] # Return only the potentially usable second column


    # --- Final Conversion Attempt for non-processed or successfully stacked data ---
    if not processed or (processed and data_X.ndim == 2 and data_X.shape[1] > 0): # Avoid trying on 1D object arrays
        try:
            # Only attempt conversion if data looks like a standard 2D numeric array
            if data_X.dtype != 'object':
                 data_X = data_X.astype(np.float32) # Convert to float32 here if possible
                 print("Converted final features to float32.")
            else:
                 print("Final features remain dtype 'object'. Needs handling before tensor conversion.")
        except ValueError:
            print("Warning: Could not convert final features to float32. Check data types in pkl file or stacking result.")

    print(f"Final check before return: Labels shape {data_Y.shape}, Features shape {data_X.shape}, Features dtype {data_X.dtype}")
    return data_Y, data_X