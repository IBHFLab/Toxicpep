import csv
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc)
import pandas as pd
import numpy as np
from util.Train_util_XU import load_data_pkl,load_data_csv,get_CM
from models.Model_XU import CNN_Model, BiGRU_Model, AttTransform_Model, Mamba_Model, CustomDataset, OptimizedCNN_Model,OptimizedBiGRU_Model,AdvancedCNN_Model
import os
from MLP_train import MLP


def main(model_type,feature_type,input_channel):

#XU
    y,t5 = load_data_pkl ("./data/XU/protein_embeddings_XU_AMP_test.pkl",
            "./data/XU/protein_embeddings_XU_nonAMP_test.pkl")

#XU
    _, esm = load_data_csv('./data/XU/XU_AMP_test_processed.csv',
                           './data/XU/XU_nonAMP_test_processed.csv')#+++

    t5_esm = np.hstack((t5,esm))

    tag = 'toxic'
    model_path = './data/XU/OpCNN_t5_esm1b_results/model.pt'
    test_roc_path = './data/XU/OpCNN_t5_esm1b_test_results/test_roc.csv'
    test_pr_path = './data/XU/OpCNN_t5_esm1b_test_results/test_pr.csv'

    y, t5 = pd.DataFrame(y), pd.DataFrame(t5)
    esm = pd.DataFrame(esm)

    print(f't5.shape:{t5.shape}')
    print(f'y.shape:{y.shape}')
    print(f'esm.shape:{esm.shape}')

    batch_size = 64

    features = {
        "t5": t5,  # 转换为NumPy数组
        "esm": esm.values,
        "t5_esm": t5_esm,
    }

    Test_feature = features[feature_type]
    Test_y= y
    test_dataset = CustomDataset(Test_feature, Test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    models = {
        "CNN": CNN_Model,
        "Transformer": AttTransform_Model,
        "BiGRU": BiGRU_Model,
        "Mamba": Mamba_Model,
        "OptimizedCNN": OptimizedCNN_Model,
        "OptimizedBiGRU": OptimizedBiGRU_Model,
        "AdvancedCNN":AdvancedCNN_Model,
        "MLP": MLP  # 添加MLP到模型字典
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')

    # 根据model_type实例化模型
    if model_type == "Mamba":
        model = models[model_type](input_channel, d_model=217, d_state=27, d_conv=4, expand=3).to(device)
    elif model_type == "MLP":
        model = models[model_type](input_channel).to(device)  # MLP使用input_channel作为input_features
    else:
        model = models[model_type](input_channel).to(device)

    model.load_state_dict(torch.load(model_path))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model.eval()
    model.to(device)
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        all_auc = []

        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(dim=-1)

            # 根据模型类型调整数据维度
            if model_type in ["CNN", "OptimizedCNN", "AdvancedCNN"]:
                data = data.unsqueeze(1)  # CNN需要额外维度
            elif model_type == "MLP":
                data = data.view(data.size(0), -1)  # MLP需要展平输入

            final_output = model(data)
            # if model_type != "MLP":  # MLP已在forward中squeeze，其他模型需要手动squeeze
            #     final_output = final_output.squeeze(1)
            scores = final_output.tolist()
            all_auc.extend(scores)
            final_output = (final_output.data > 0.5).int()
            all_labels.extend(labels.tolist())
            all_predictions.extend(final_output.tolist())

        acc, sn, sp, mcc, pr, f1 = get_CM(all_labels, all_predictions)

        test_accuracy = acc
        test_precision = pr
        test_auc_roc = roc_auc_score(all_labels, all_auc)
        test_recall = recall_score(all_labels, all_predictions)
        test_f1 = f1
        precision, recall, _ = precision_recall_curve(all_labels, all_auc)
        fpr, tpr, _ = roc_curve(all_labels, all_auc)
        test_pr_auc = auc(recall, precision)
        test_roc_curve_data = pd.DataFrame({'FPR': fpr, 'TPR': tpr})
        test_roc_curve_data.to_csv(test_roc_path, index=False)
        test_pr_curve_data = pd.DataFrame({'Precision': precision, 'Recall': recall})
        test_pr_curve_data.to_csv(test_pr_path, index=False)

        print(
            f"acc:{test_accuracy:.4f} pr:{test_precision:.4f} recall{test_recall:.4f} "
            f"f1:{test_f1:.4f} roc:{test_auc_roc:.4f} auc:{test_pr_auc:.4f} "
            f"mcc:{mcc:.4f}sn:{sn:.4f}sp:{sp:.4f}")

        '''save acc'''

        csv_file = './data/XU/OpCNN_t5_esm1b_test_results/test_results.csv'
        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='') as file1:
            writer = csv.writer(file1, dialect='excel')
            # 如果文件不存在，则先写入表头
            if not file_exists:
                writer.writerow(["Tag", "Accuracy", "Sensitivity", "Specificity", "MCC", "F1", "Recall", "ROC_AUC", "PR_AUC"])
            label = tag + ': test'
            writer.writerow([label, acc, sn, sp, mcc, f1, test_recall, test_auc_roc, test_pr_auc])

if __name__ == '__main__':
    model = 'OptimizedCNN'
    feature = 't5_esm'
    input_channel = 2302 # 使用全局变量 t5_esm 的维度
    main(model, feature, input_channel)
