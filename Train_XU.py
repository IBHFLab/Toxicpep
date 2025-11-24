import csv
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.Train_util_wo_XU import load_data_pkl,load_data_csv,get_CM
from models.Model_wo_XU import CNN_Model,BiGRU_Model,AttTransform_Model,CustomDataset, OptimizedCNN_Model,OptimizedBiGRU_Model,AdvancedCNN_Model, Mamba_Model
import os


def main(model_type,feature_type,input_channel):

#XU
    y, t5 = load_data_pkl("./data/XU/protein_embeddings_XU_train_positive.pkl",
                          "./data/XU/protein_embeddings_XU_train_negative.pkl")

    _,esm = load_data_csv("./data/XU/XU_train_positive_processed.csv",
                          "./data/XU/XU_train_negative_processed.csv")

    t5_esm = np.hstack((t5,esm))
    assert len(y) == t5_esm.shape[0], "特征和标签样本数量不匹配"

    tag = 'toxic'
    model_path = './data/XU/OpCNN_t5_esm1b_results/model.pt'#+++
    train_roc_path = './data/XU/OpCNN_t5_esm1b_results/train_roc.csv'#+++
    train_pr_path = './data/XU/OpCNN_t5_esm1b_results/train_pr.csv'#+++

    y, t5 = pd.DataFrame(y), pd.DataFrame(t5)
    esm = pd.DataFrame(esm)

    print(f't5.shape:{t5.shape}')
    print(f'y.shape:{y.shape}')
    print(f'esm.shape:{esm.shape}')

    num_epochs = 100
    batch_size = 64
    num_folds = 5
    num_t = 1
    al_acc = 0
    all_acc = 0
    all_sn = 0
    al_pr = 0
    all_pr = 0
    al_sn = 0
    all_sp = 0
    al_sp = 0
    all_mcc = 0
    al_mcc = 0
    all_f1 = 0
    al_f1 = 0
    all_recall = 0
    al_recall = 0
    all_train_roc = 0
    all_val_roc = 0
    all_val_prauc = 0
    all_train_prauc = 0
    num_comp_time = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')

    tprs = []
    fprs = []
    precisions = []
    recalls = []
    best_acc = 0
    best_acc_model = None
    mean_fpr_linspace = np.linspace(0, 1, 100)
    i = 0
    features = {
        "t5": t5.values,  # 转换为NumPy数组
        "esm": esm.values,  # 转换为NumPy数组
        "t5_esm": t5_esm,
    }
    # 修改：使用特征数据创建KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    results_path = './data/XU/OpCNN_t5_esm1b_results/results.csv'#+++
    # 定义列名
    columns = ['label', 'acc', 'sn', 'sp', 'mcc', 'f1', 'recall', 'roc', 'pr_auc']
    # 检查文件是否存在，如果不存在，写入列名
    if not os.path.exists(results_path):
        with open(results_path, 'w', newline='') as file1:
            content1 = csv.writer(file1, dialect='excel')
            content1.writerow(columns)
    for fold, (train_index, val_index) in enumerate(kf.split(features[feature_type])):
        n = 0
        print(f"Fold: {fold + 1}")
        i += 1

        # 使用正确的索引访问特征和标签
        Train_feature = features[feature_type][train_index]  # 直接使用NumPy索引
        Train_y = y.iloc[train_index] if hasattr(y, 'iloc') else y[train_index]  # 根据y的类型选择索引方式

        val_feature = features[feature_type][val_index]  # 直接使用NumPy索引
        val_y = y.iloc[val_index] if hasattr(y, 'iloc') else y[val_index]  # 根据y的类型选择索引方式

        train_dataset = CustomDataset(Train_feature, Train_y)
        val_dataset = CustomDataset(val_feature, val_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        models = {
            "CNN": CNN_Model,
            "Transformer": AttTransform_Model,
            "BiGRU": BiGRU_Model,
            "Mamba": Mamba_Model,
            "OptimizedCNN":OptimizedCNN_Model,
            "OptimizedBiGRU":OptimizedBiGRU_Model,
            "AdvancedCNN":AdvancedCNN_Model
        }

        Model = models[model_type](input_channel)
        Model.to(device)
        criterion = nn.BCELoss()
        # criterion = nn.HingeEmbeddingLoss()

        optimizer = optim.Adam(Model.parameters(), lr=0.00001)

        final_output_list = []

        for n in range(num_t):
            losses = []
            lossess = []
            acces = []
            access = []
            for epoch in range(num_epochs):
                al_predictions = []
                al_labels = []
                al_auc = []
                Model.train()
                for data, labels in train_loader:
                    data = data.to(device)
                    if model_type in ["CNN", "OptimizedCNN","AdvancedCNN"]:
                        data = data.unsqueeze(1)
                    labels = labels.to(device)
                    labels = labels.squeeze(dim=-1)
                    # print("labels unique values:", torch.unique(labels))
                    optimizer.zero_grad()
                    final_output = Model(data)
                    loss = criterion(final_output, labels)
                    loss.backward()
                    optimizer.step()

                    scores = final_output.tolist()
                    al_auc.extend(scores)
                    final_output = (final_output.data > 0.7).int()
                    al_labels.extend(labels.tolist())
                    al_predictions.extend(final_output.tolist())

                accc, snn, spp, mccc, prr, f11 = get_CM(al_labels, al_predictions)
                train_roc = roc_auc_score(al_labels, al_auc)
                all_train_roc += train_roc
                al_mcc += mccc
                train_recall = recall_score(al_labels, al_predictions)
                precisionn, recalll, _ = precision_recall_curve(al_labels, al_auc)
                fprr, tprr, _ = roc_curve(al_labels, al_auc)
                train_pr_auc = auc(recalll, precisionn)
                all_train_prauc += train_pr_auc
                
                train_f1 = f11
                al_sp += spp
                al_pr += prr
                al_sn += snn
                al_f1 += train_f1
                al_recall+=train_recall
                al_acc += accc
                access.append(accc)
                loss = loss.detach().cpu().numpy().astype(np.float64)
                losses.append(loss)
                
                all_predictions = []
                all_labels = []
                all_auc = []

                Model.eval()
                for data, labels in val_loader:
                    if model_type in ["CNN", "OptimizedCNN","AdvancedCNN"]:
                        data = data.unsqueeze(1).to(device)

                    labels = labels.to(device)
                    labels = labels.squeeze(dim=-1)
                    optimizer.zero_grad()
                    final_output = Model(data)
                    loss = criterion(final_output, labels)
                    # final_output = final_output.squeeze(1)
                    scores = final_output.tolist()
                    all_auc.extend(scores)
                    final_output = (final_output.data > 0.7).int()
                    all_labels.extend(labels.tolist())
                    all_predictions.extend(final_output.tolist())
                    
                losss = loss.detach().cpu().numpy().astype(np.float64)
                lossess.append(losss)

                acc, sn, sp, mcc, pr, f1 = get_CM(all_labels, all_predictions)
                acces.append(acc)
                val_accuracy = acc
                val_precision = pr

                val_roc = roc_auc_score(all_labels, all_auc)
                val_recall = recall_score(all_labels, all_predictions)
                # print(f"sn: {sn}, recall: {val_recall}")
                val_f1 = f1
                precision, recall, _ = precision_recall_curve(all_labels, all_auc)
                fpr, tpr, _ = roc_curve(all_labels, all_auc)
                val_pr_auc = auc(recall, precision)
                num_samples = 100
                precision_sampled = np.linspace(0, 1, num_samples)
                recall_sampled = np.interp(precision_sampled, precision, recall)
                fpr_sampled = np.linspace(0, 1, num_samples)
                tpr_sampled = np.interp(fpr_sampled, fpr, tpr)

                fprs.append(fpr_sampled)
                tprs.append(tpr_sampled)
                precisions.append(precision_sampled)
                recalls.append(recall_sampled)

                all_acc += acc
                all_sp += sp
                all_sn += sn
                all_pr += pr
                all_mcc += mcc
                all_f1 += f1
                all_recall+=val_recall
                all_val_roc += val_roc
                all_val_prauc += val_pr_auc
                num_comp_time += 1
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_acc_model = Model.state_dict().copy()

                # 保存单次 fold 的结果
                with open(results_path, 'a', newline='') as file1:
                    content1 = csv.writer(file1, dialect='excel')
                    label = tag + ': ' + str(i) + 'k'
                    content1.writerow([label, acc, sn, sp, mcc, f1, val_recall, val_roc, val_pr_auc])

    # 保存所有 fold 的平均结果
    with open(results_path, 'a', newline='') as file1:
        content1 = csv.writer(file1, dialect='excel')
        label = tag
        content1.writerow([label, all_acc / num_comp_time, all_sn / num_comp_time,
                           all_sp / num_comp_time, all_mcc / num_comp_time, all_f1 / num_comp_time,all_recall/num_comp_time,
                           all_val_roc / num_comp_time, all_val_prauc / num_comp_time])

    mean_precision = np.mean(precisions, axis=0)
    mean_recall = np.mean(recalls, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    mean_tpr = np.mean(tprs, axis=0)
    print(
        f"train_acc:{al_acc / num_comp_time:.4f} train_pr:{al_pr / num_comp_time:.4f} train_recall{al_recall/num_comp_time:.4f} "
        f"train_f1:{al_f1 / num_comp_time:.4f} train_roc:{all_train_roc / num_comp_time:.4f} train_auc:{all_train_prauc/num_comp_time:.4f} "
        f"train_mcc:{al_mcc / num_comp_time:.4f}train_sn:{al_sn / num_comp_time:.4f}train_sp:{al_sp / num_comp_time:.4f}")

    val_pr_curve_data = pd.DataFrame({'Precision': mean_precision, 'Recall': mean_recall})
    val_pr_curve_data.to_csv(train_roc_path, index=False)
    val_roc_curve_data = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
    val_roc_curve_data.to_csv(train_pr_path, index=False)
    print("")
    print(
        f"val_acc:{all_acc / num_comp_time:.4f} val_pr:{all_pr / num_comp_time:.4f} val_recall{all_recall / num_comp_time:.4f} "
        f"val_f1:{all_f1 / num_comp_time:.4f} val_roc:{all_val_roc / num_comp_time:.4f} val_auc:{all_val_prauc/num_comp_time:.4f} "
        f"val_mcc:{all_mcc / num_comp_time:.4f}val_sn:{all_sn / num_comp_time:.4f}val_sp:{all_sp / num_comp_time:.4f}")

    torch.save(best_acc_model, model_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), losses, label='Loss', color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), acces, label='Accuracy', color='green')
    plt.plot(range(num_epochs), access, label='Accuracy', color='green')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_loss_curve.png')


if __name__ == '__main__':
    model = 'OptimizedCNN'
    feature = 't5_esm'
    input_channel = 2302
    main(model,feature,input_channel)