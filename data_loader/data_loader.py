import pandas as pd
import numpy as np
import os
from util.Train_util import load_data_pkl

# 获取当前脚本的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_all_features():
    """加载所有特征数据"""
    features_dict = {}
    labels_dict = {}

    try:
        y2, t5 = load_data_pkl(
            os.path.join(BASE_DIR,'data_loader/Prott5/test_pos.pkl'),
            os.path.join(BASE_DIR,'data_loader/Prott5/test_neg.pkl'))
        features_dict["t5"] = pd.DataFrame(t5)
        labels_dict["t5"] = pd.DataFrame(y2)
        print(f'y2.shape:{y2.shape}')
        print(f't5.shape:{t5.shape}')
    except Exception as e:
        print(f"加载ProtT5特征时出错: {e}")

    # 加载ESM1b特征
    try:
        y3, esm = load_data_pkl(
            os.path.join(BASE_DIR,'data_loader/esm1b_t33_650M_UR50S/test_pos.pkl'),
            os.path.join(BASE_DIR,'data_loader/esm1b_t33_650M_UR50S/test_neg.pkl'))
        features_dict["esm"] = pd.DataFrame(esm)
        labels_dict["esm"] = pd.DataFrame(y3)
        print(f'y3.shape:{y3.shape}')
        print(f'esm.shape:{esm.shape}')
    except Exception as e:
        print(f"加载ESM1b特征时出错: {e}")

    return features_dict, labels_dict

def generate_feature_combinations(features_dict):
    """生成所有可能的特征组合"""
    feature_combinations = {}

    # 单一特征
    for name, feature in features_dict.items():
        feature_combinations[name] = feature.values

    # 两特征组合
    if "t5" in features_dict and "dr" in features_dict:
        feature_combinations["t5_dr"] = np.hstack((features_dict["t5"].values, features_dict["dr"].values))

    if "t5" in features_dict and "esm" in features_dict:
        feature_combinations["t5_esm"] = np.hstack((features_dict["t5"].values, features_dict["esm"].values))

    if "t5" in features_dict and "esm2_t6" in features_dict:
        feature_combinations["t5_esm2_t6"] = np.hstack((features_dict["t5"].values, features_dict["esm2_t6"].values))

    if "t5" in features_dict and "esm2_t12" in features_dict:
        feature_combinations["t5_esm2_t12"] = np.hstack((features_dict["t5"].values, features_dict["esm2_t12"].values))

    if "t5" in features_dict and "esm2_t30" in features_dict:
        feature_combinations["t5_esm2_t30"] = np.hstack((features_dict["t5"].values, features_dict["esm2_t30"].values))

    if "t5" in features_dict and "esm2_t36" in features_dict:
        feature_combinations["t5_esm2_t36"] = np.hstack((features_dict["t5"].values, features_dict["esm2_t36"].values))

    if "dr" in features_dict and "esm" in features_dict:
        feature_combinations["dr_esm"] = np.hstack((features_dict["dr"].values, features_dict["esm"].values))

    if "dr" in features_dict and "esm2_t6" in features_dict:
        feature_combinations["dr_esm2_t6"] = np.hstack((features_dict["dr"].values, features_dict["esm2_t6"].values))

    if "dr" in features_dict and "esm2_t12" in features_dict:
        feature_combinations["dr_esm2_t12"] = np.hstack((features_dict["dr"].values, features_dict["esm2_t12"].values))

    if "dr" in features_dict and "esm2_t30" in features_dict:
        feature_combinations["dr_esm2_t30"] = np.hstack((features_dict["dr"].values, features_dict["esm2_t30"].values))

    if "dr" in features_dict and "esm2_t36" in features_dict:
        feature_combinations["dr_esm2_t36"] = np.hstack((features_dict["dr"].values, features_dict["esm2_t36"].values))

    # 三特征组合
    if "t5" in features_dict and "esm" in features_dict and "dr" in features_dict:
        feature_combinations["t5_esm_dr"] = np.hstack((
            features_dict["t5"].values,
            features_dict["esm"].values,
            features_dict["dr"].values
        ))

    if "t5" in features_dict and "dr" in features_dict and "esm2_t36" in features_dict:
        feature_combinations["t5_dr_esm2_t36"] = np.hstack((
            features_dict["t5"].values,
            features_dict["dr"].values,
            features_dict["esm2_t36"].values
        ))

    if "t5" in features_dict and "dr" in features_dict and "esm2_t30" in features_dict:
        feature_combinations["t5_dr_esm2_t30"] = np.hstack((
            features_dict["t5"].values,
            features_dict["dr"].values,
            features_dict["esm2_t30"].values
        ))

    # 验证每个特征组合的有效性
    for name, feature_data in feature_combinations.items():
        if "t5" in features_dict:
            assert len(features_dict["t5"]) == feature_data.shape[0], f"特征{name}和标签样本数量不匹配"

    return feature_combinations

