import csv
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import (recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc)
import pandas as pd
import numpy as np
from util.Train_util import load_data_pkl, get_CM
from models.Model import OptimizedCNN_Model, CNN_Model, CustomDataset
import os
import shutil
import argparse
import random

seed = 42  # 可选择任意固定值
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)  # 如果使用 GPU
torch.backends.cudnn.deterministic = True  # 确保 CuDNN 行为确定
torch.backends.cudnn.benchmark = False  # 禁用 CuDNN 的优化以保证一致性

def load_all_features():
    """加载所有特征数据"""
    features_dict = {}
    labels_dict = {}

    # 加载ProtT5特征
    try:
        y2, t5 = load_data_pkl(
            './data_loader/Prott5/train_pos.pkl',
            './data_loader/Prott5/train_neg.pkl')
        features_dict["t5"] = pd.DataFrame(t5)
        labels_dict["t5"] = pd.DataFrame(y2)
        print(f'y2.shape:{y2.shape}')
        print(f't5.shape:{t5.shape}')
    except Exception as e:
        print(f"加载ProtT5特征时出错: {e}")

    # 加载ESM1b特征
    try:
        y3, esm = load_data_pkl(
            './data_loader/esm1b_t33_650M_UR50S/train_pos.pkl',
            './data_loader/esm1b_t33_650M_UR50S/train_neg.pkl')
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

def setup_output_paths(model_type, feature_name, target_name="toxic"):
    """创建并返回所有输出文件路径"""
    base_folder = os.path.join('./position', f"{model_type}_{feature_name}_results")
    result_folder = os.path.join(base_folder, target_name)
    os.makedirs(result_folder, exist_ok=True)

    paths = {
        "model": os.path.join(result_folder, 'model.pt'),
        "train_roc": os.path.join(result_folder, 'train_roc.csv'),
        "train_pr": os.path.join(result_folder, 'train_pr.csv'),
        "results": os.path.join(result_folder, 'results.csv'),
        "scaler": os.path.join(result_folder, 'scaler.pkl'),
        "train_acc": os.path.join(result_folder, 'best_fold_train_accuracies.npy'),
        "val_acc": os.path.join(result_folder, 'best_fold_val_accuracies.npy'),
        "train_loss": os.path.join(result_folder, 'best_fold_train_losses.npy'),
        "val_loss": os.path.join(result_folder, 'best_fold_val_losses.npy')
    }

    return paths


def train_model_fold(model, train_loader, val_loader, optimizer, scheduler, device,
                     num_epochs, model_type, patience, min_delta, lr_scheduler_type):
    """对单个折执行模型训练"""
    criterion = nn.BCELoss()

    # 初始化追踪变量
    best_val_loss = float('inf')
    wait = 0
    best_model_state = None
    early_stopped = False

    # 初始化最佳性能指标
    best_fold_epoch = -1
    best_fold_acc = 0
    best_fold_sp = 0
    best_fold_sn = 0
    best_fold_mcc = 0
    best_fold_f1 = 0
    best_fold_recall = 0
    best_fold_roc = 0
    best_fold_prauc = 0

    # 记录训练历史
    current_fold_train_losses = []
    current_fold_val_losses = []
    current_fold_train_accuracies = []
    current_fold_val_accuracies = []

    # 记录ROC和PR曲线数据
    tprs = []
    fprs = []
    precisions = []
    recalls = []

    # 执行训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for data, labels in train_loader:
            # 根据模型类型准备数据
            data = data.to(device)
            if model_type in ["CNN", "OptimizedCNN"]:
                data = data.unsqueeze(1)
            labels = labels.to(device).squeeze(dim=-1)

            # 标准模型训练
            optimizer.zero_grad()
            final_output = model(data)
            loss = criterion(final_output, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        # 计算平均训练损失
        avg_epoch_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
        current_fold_train_losses.append(avg_epoch_train_loss)

        # 计算训练准确率
        model.eval()
        train_all_preds_acc = []
        train_all_labels_acc = []

        with torch.no_grad():
            for data, labels in train_loader:
                data_acc = data.to(device)
                if model_type in ["CNN", "OptimizedCNN"]:
                    data_acc = data_acc.unsqueeze(1)
                labels_acc_cpu = labels.squeeze(dim=-1).cpu().tolist()

                final_output = model(data_acc)

                preds_acc = (final_output.cpu().data > 0.5).int().tolist()
                train_all_preds_acc.extend(preds_acc)
                train_all_labels_acc.extend(labels_acc_cpu)

        if train_all_labels_acc:
            acc_train, _, _, _, _, _ = get_CM(train_all_labels_acc, train_all_preds_acc)
            current_fold_train_accuracies.append(acc_train)
        else:
            current_fold_train_accuracies.append(0.0)

        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        all_predictions = []
        all_labels = []
        all_scores = []

        with torch.no_grad():
            for data, labels in val_loader:
                data_val = data.to(device)
                if model_type in ["CNN", "OptimizedCNN"]:
                    data_val = data_val.unsqueeze(1)
                labels_val = labels.to(device).squeeze(dim=-1).float()


                final_output = model(data_val)

                val_loss = criterion(final_output, labels_val)
                epoch_val_loss += val_loss.item()
                num_val_batches += 1

                scores = final_output.cpu().tolist()
                all_scores.extend(scores)
                predictions = (final_output.cpu().data > 0.5).int().tolist()
                all_predictions.extend(predictions)
                all_labels.extend(labels_val.cpu().tolist())

        # 计算平均验证损失
        avg_epoch_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
        current_fold_val_losses.append(avg_epoch_val_loss)

        # 计算评估指标
        acc, sn, sp, mcc, pr, f1 = get_CM(all_labels, all_predictions)
        current_fold_val_accuracies.append(acc)
        val_recall_metric = recall_score(all_labels, all_predictions)

        # 计算ROC和PR曲线
        try:
            val_roc = roc_auc_score(all_labels, all_scores)
            precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
            fpr_curve, tpr_curve, _ = roc_curve(all_labels, all_scores)
            val_pr_auc = auc(recall_curve, precision_curve)
        except ValueError:
            print(f"Warning: ROC/PR AUC calculation failed. Assigning 0.")
            val_roc = 0.0
            val_pr_auc = 0.0
            precision_curve, recall_curve = np.array([0, 1]), np.array([1, 0])
            fpr_curve, tpr_curve = np.array([0, 1]), np.array([0, 1])

        # 曲线采样
        num_samples = 100
        precision_sampled_linspace = np.linspace(0, 1, num_samples)
        fpr_sampled_linspace = np.linspace(0, 1, num_samples)

        if len(precision_curve) > 1 and len(recall_curve) > 1:
            sort_indices = np.argsort(precision_curve)
            recall_sampled = np.interp(precision_sampled_linspace, precision_curve[sort_indices],
                                       recall_curve[sort_indices])
        else:
            recall_sampled = np.zeros(num_samples)

        if len(fpr_curve) > 1 and len(tpr_curve) > 1:
            tpr_sampled = np.interp(fpr_sampled_linspace, fpr_curve, tpr_curve)
        else:
            tpr_sampled = np.linspace(0, 1, num_samples)

        # 添加采样曲线
        fprs.append(fpr_sampled_linspace)
        tprs.append(tpr_sampled)
        precisions.append(precision_sampled_linspace)
        recalls.append(recall_sampled)

        # 输出当前训练状态
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_epoch_train_loss:.4f} | Train Acc: {acc_train:.4f} | "
              f"Val Loss: {avg_epoch_val_loss:.4f} | Val Acc: {acc:.4f} | "
              f"Val ROC: {val_roc:.4f} | Val PR AUC: {val_pr_auc:.4f}")

        # 更新学习率调度器
        if scheduler:
            if lr_scheduler_type == 'plateau':
                scheduler.step(avg_epoch_val_loss)
            else:
                scheduler.step()

        # 检查早停条件
        if avg_epoch_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_epoch_val_loss
            wait = 0
            best_model_state = model.state_dict().copy()

            # 更新最佳指标
            best_fold_epoch = epoch + 1
            best_fold_acc = acc
            best_fold_sp = sp
            best_fold_sn = sn
            best_fold_mcc = mcc
            best_fold_f1 = f1
            best_fold_recall = val_recall_metric
            best_fold_roc = val_roc
            best_fold_prauc = val_pr_auc

            print(f"    -> New best val loss: {best_val_loss:.4f}. Model state updated.")
        else:
            wait += 1
            if wait >= patience:
                print(f"    -> Early stopping triggered after {patience} epochs.")
                early_stopped = True
                break

    # 返回训练结果
    result = {
        "best_model_state": best_model_state,
        "best_epoch": best_fold_epoch,
        "best_metrics": {
            "acc": best_fold_acc,
            "sp": best_fold_sp,
            "sn": best_fold_sn,
            "mcc": best_fold_mcc,
            "f1": best_fold_f1,
            "recall": best_fold_recall,
            "roc": best_fold_roc,
            "prauc": best_fold_prauc
        },
        "history": {
            "train_losses": current_fold_train_losses,
            "val_losses": current_fold_val_losses,
            "train_accuracies": current_fold_train_accuracies,
            "val_accuracies": current_fold_val_accuracies
        },
        "curves": {
            "tprs": tprs,
            "fprs": fprs,
            "precisions": precisions,
            "recalls": recalls
        }
    }

    return result


def train_model(feature_type, model_type, target_name="toxic", lr_scheduler_type="plateau"):
    """训练模型的主函数"""
    print(f"开始训练模型 - 特征类型: {feature_type}, 模型类型: {model_type}")

    # 加载所有特征数据
    features_dict, labels_dict = load_all_features()

    # 生成特征组合
    feature_combinations = generate_feature_combinations(features_dict)

    # 检查选择的特征是否存在
    if feature_type not in feature_combinations:
        print(f"错误: 特征类型 '{feature_type}' 不可用。可用特征类型: {list(feature_combinations.keys())}")
        return

    # 设置输出路径
    paths = setup_output_paths(model_type, feature_type, target_name)

    # 创建结果文件
    if not os.path.exists(paths["results"]):
        with open(paths["results"], 'w', newline='') as file1:
            content1 = csv.writer(file1, dialect='excel')
            columns = ['label', 'acc', 'sn', 'sp', 'mcc', 'f1', 'recall', 'roc', 'pr_auc']
            content1.writerow(columns)

    # 配置训练参数
    num_epochs = 120
    batch_size = 16
    num_folds = 5
    patience = 10
    min_delta = 0.01
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    # 选择要使用的特征和标签
    selected_feature_data = feature_combinations[feature_type]
    input_channel = selected_feature_data.shape[1]

    # 使用t5对应的标签（假设所有特征的标签都是一致的）
    y = labels_dict["t5"] if "t5" in labels_dict else list(labels_dict.values())[0]

    # 模型定义
    models_dict = {
        "CNN": CNN_Model,
        "OptimizedCNN": OptimizedCNN_Model,
    }

    # K折交叉验证
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # 用于记录所有折的结果
    fold_results_list = []

    # 全局性能指标
    all_acc = 0
    all_sn = 0
    all_sp = 0
    all_mcc = 0
    all_f1 = 0
    all_recall = 0
    all_val_roc = 0
    all_val_prauc = 0
    num_comp_time = 0

    # 记录所有折的数据
    all_tprs = []
    all_fprs = []
    all_precisions = []
    all_recalls = []
    all_folds_train_losses = []
    all_folds_val_losses = []
    all_folds_train_accuracies = []
    all_folds_val_accuracies = []

    # 执行K折训练
    for fold, (train_index, val_index) in enumerate(kf.split(selected_feature_data, y)):
        print(f"\n========== 执行第 {fold + 1} 折 ==========")

        # 准备训练和验证数据
        train_feature = selected_feature_data[train_index]
        train_y = y.iloc[train_index] if hasattr(y, 'iloc') else y[train_index]

        val_feature = selected_feature_data[val_index]
        val_y = y.iloc[val_index] if hasattr(y, 'iloc') else y[val_index]

        train_dataset = CustomDataset(train_feature, train_y)
        val_dataset = CustomDataset(val_feature, val_y)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)


        if model_type == "OptimizedCNN":
            model = OptimizedCNN_Model(
                input_channels=input_channel,
                hidden_dim=512,
                dropout_rate=0.4,
                activation='gelu'
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.00001)
        else:
            model = models_dict[model_type](input_channel).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.000001, weight_decay=0.00001)

        # 设置学习率调度器
        if lr_scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif lr_scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif lr_scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        else:
            scheduler = None

        # 执行训练
        fold_result = train_model_fold(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            num_epochs=num_epochs,
            model_type=model_type,
            patience=patience,
            min_delta=min_delta,
            lr_scheduler_type=lr_scheduler_type
        )

        # 保存此折的最佳模型
        best_model_state = fold_result["best_model_state"]
        if best_model_state:
            fold_model_path = os.path.join(os.path.dirname(paths["model"]), f'model_fold_{fold + 1}_best.pt')
            checkpoint = {
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, fold_model_path)
            print(f"保存第 {fold + 1} 折最佳模型到 {fold_model_path}")

            # 记录此折的结果
            fold_results_list.append({
                'fold': fold + 1,
                'best_roc': fold_result["best_metrics"]["roc"],
                'model_path': fold_model_path
            })

            # 累加指标
            metrics = fold_result["best_metrics"]
            all_acc += metrics["acc"]
            all_sp += metrics["sp"]
            all_sn += metrics["sn"]
            all_mcc += metrics["mcc"]
            all_f1 += metrics["f1"]
            all_recall += metrics["recall"]
            all_val_roc += metrics["roc"]
            all_val_prauc += metrics["prauc"]
            num_comp_time += 1
        else:
            print(f"警告: 第 {fold + 1} 折没有观察到性能提升，未保存模型。")

        # 记录曲线数据
        all_tprs.extend(fold_result["curves"]["tprs"])
        all_fprs.extend(fold_result["curves"]["fprs"])
        all_precisions.extend(fold_result["curves"]["precisions"])
        all_recalls.extend(fold_result["curves"]["recalls"])

        # 记录损失和准确率历史
        all_folds_train_losses.append(fold_result["history"]["train_losses"])
        all_folds_val_losses.append(fold_result["history"]["val_losses"])
        all_folds_train_accuracies.append(fold_result["history"]["train_accuracies"])
        all_folds_val_accuracies.append(fold_result["history"]["val_accuracies"])

    # 计算并保存平均结果
    print("\n========== 交叉验证总结 ==========")
    if num_comp_time > 0:
        avg_acc = all_acc / num_comp_time
        avg_sn = all_sn / num_comp_time
        avg_sp = all_sp / num_comp_time
        avg_mcc = all_mcc / num_comp_time
        avg_f1 = all_f1 / num_comp_time
        avg_recall = all_recall / num_comp_time
        avg_roc = all_val_roc / num_comp_time
        avg_prauc = all_val_prauc / num_comp_time

        print(f"平均准确率: {avg_acc:.4f}")
        print(f"平均灵敏度: {avg_sn:.4f}")
        print(f"平均特异性: {avg_sp:.4f}")
        print(f"平均MCC: {avg_mcc:.4f}")
        print(f"平均F1: {avg_f1:.4f}")
        print(f"平均召回率: {avg_recall:.4f}")
        print(f"平均ROC曲线下面积: {avg_roc:.4f}")
        print(f"平均PR曲线下面积: {avg_prauc:.4f}")

        # 保存平均结果到CSV
        with open(paths["results"], 'a', newline='') as file1:
            content1 = csv.writer(file1, dialect='excel')
            label = f"{target_name}: 平均性能"
            content1.writerow([label, avg_acc, avg_sn, avg_sp, avg_mcc, avg_f1,
                               avg_recall, avg_roc, avg_prauc])
    else:
        print("警告: 训练过程中没有保存有效模型。请检查训练参数。")

    # 保存平均ROC和PR曲线
    if all_tprs and all_fprs:
        num_samples = len(all_fprs[0])
        mean_fpr = np.linspace(0, 1, num_samples)
        mean_tpr = np.mean(all_tprs, axis=0)

        val_roc_curve_data = pd.DataFrame({'FPR': mean_fpr, 'TPR': mean_tpr})
        val_roc_curve_data.to_csv(paths["train_roc"], index=False)
        print(f"保存平均ROC曲线数据到 {paths['train_roc']}")

    if all_recalls and all_precisions:
        num_samples = len(all_precisions[0])
        mean_precision = np.linspace(0, 1, num_samples)
        mean_recall = np.mean(all_recalls, axis=0)

        val_pr_curve_data = pd.DataFrame({'Precision': mean_precision, 'Recall': mean_recall})
        val_pr_curve_data.to_csv(paths["train_pr"], index=False)
        print(f"保存平均PR曲线数据到 {paths['train_pr']}")

    # 选择并保存整体最佳模型
    print("\n========== 选择并保存整体最佳模型 ==========")
    overall_best_roc = -1.0
    overall_best_model_path = None
    best_fold_info = None

    if not fold_results_list:
        print("警告: 没有记录任何折的结果。无法确定整体最佳模型。")
    else:
        # 找出具有最高ROC值的折
        for result in fold_results_list:
            if result.get('model_path') and result['best_roc'] > overall_best_roc:
                overall_best_roc = result['best_roc']
                overall_best_model_path = result['model_path']
                best_fold_info = result

        if overall_best_model_path and os.path.exists(overall_best_model_path):
            print(f"找到整体最佳模型: 第 {best_fold_info['fold']} 折, ROC: {overall_best_roc:.4f}")
            final_model_target_path = paths["model"]

            try:
                # 复制最佳折的模型文件到最终目标路径
                shutil.copy2(overall_best_model_path, final_model_target_path)
                print(f"已复制最佳模型到: {final_model_target_path}")

                # 删除中间折模型（保留最佳折模型）
                print("删除中间折模型（保留整体最佳模型）...")
                deleted_count = 0
                kept_count = 0

                for result in fold_results_list:
                    fold_model_file = result.get('model_path')
                    if fold_model_file and os.path.exists(fold_model_file):
                        if fold_model_file != overall_best_model_path:
                            try:
                                os.remove(fold_model_file)
                                deleted_count += 1
                            except OSError as e:
                                print(f"删除文件 {fold_model_file} 时出错: {e}")
                        else:
                            print(f"保留原始最佳折模型: {fold_model_file}")
                            kept_count += 1

                print(f"中间模型删除完成。已删除: {deleted_count}, 已保留: {kept_count}。")

            except Exception as e:
                print(f"复制最佳模型文件时出错: {e}")
                print("由于复制错误，未删除单个折模型。")
        else:
            print("警告: 无法找到或访问保存的最佳整体模型文件。")
            print("未创建最终的'model.pt'，且未删除折模型。")

    # 保存最佳折的历史数据
    print("\n========== 保存最佳折的历史数据 ==========")
    if best_fold_info:
        best_fold_index = best_fold_info['fold'] - 1  # 列表索引从0开始

        try:
            # 保存最佳折的训练损失
            if best_fold_index < len(all_folds_train_losses) and all_folds_train_losses[best_fold_index]:
                np.save(paths["train_loss"], np.array(all_folds_train_losses[best_fold_index]))
                print(f"已保存第 {best_fold_info['fold']} 折训练损失到 {paths['train_loss']}")
            else:
                print(f"警告: 未找到第 {best_fold_info['fold']} 折的训练损失历史。")

            # 保存最佳折的验证损失
            if best_fold_index < len(all_folds_val_losses) and all_folds_val_losses[best_fold_index]:
                np.save(paths["val_loss"], np.array(all_folds_val_losses[best_fold_index]))
                print(f"已保存第 {best_fold_info['fold']} 折验证损失到 {paths['val_loss']}")
            else:
                print(f"警告: 未找到第 {best_fold_info['fold']} 折的验证损失历史。")

            # 保存最佳折的训练准确率
            if best_fold_index < len(all_folds_train_accuracies) and all_folds_train_accuracies[best_fold_index]:
                np.save(paths["train_acc"], np.array(all_folds_train_accuracies[best_fold_index]))
                print(f"已保存第 {best_fold_info['fold']} 折训练准确率到 {paths['train_acc']}")
            else:
                print(f"警告: 未找到第 {best_fold_info['fold']} 折的训练准确率历史。")

            # 保存最佳折的验证准确率
            if best_fold_index < len(all_folds_val_accuracies) and all_folds_val_accuracies[best_fold_index]:
                np.save(paths["val_acc"], np.array(all_folds_val_accuracies[best_fold_index]))
                print(f"已保存第 {best_fold_info['fold']} 折验证准确率到 {paths['val_acc']}")
            else:
                print(f"警告: 未找到第 {best_fold_info['fold']} 折的验证准确率历史。")

        except IndexError:
            print(f"错误: 最佳折索引 {best_fold_index} 超出历史列表范围。")
        except Exception as e:
            print(f"保存最佳折历史数据时出错: {e}")
    else:
        print("警告: 无法保存历史数据，因为未识别出最佳折。")

    # 返回训练结果摘要
    results_summary = {
        "avg_metrics": {
            "acc": avg_acc if num_comp_time > 0 else 0,
            "sn": avg_sn if num_comp_time > 0 else 0,
            "sp": avg_sp if num_comp_time > 0 else 0,
            "mcc": avg_mcc if num_comp_time > 0 else 0,
            "f1": avg_f1 if num_comp_time > 0 else 0,
            "recall": avg_recall if num_comp_time > 0 else 0,
            "roc": avg_roc if num_comp_time > 0 else 0,
            "pr_auc": avg_prauc if num_comp_time > 0 else 0
        },
        "best_fold": best_fold_info,
        "model_path": paths["model"] if best_fold_info else None
    }

    return results_summary


def main():
    """主函数，解析命令行参数并执行训练"""
    parser = argparse.ArgumentParser(description='蛋白质特征分析与模型训练')

    # 添加参数
    parser.add_argument('--model', type=str, default='OptimizedCNN',
                        choices=['OptimizedCNN','CNN'],
                        help='要使用的模型类型')

    parser.add_argument('--feature', type=str, default='t5_esm',
                        help='要使用的特征类型或特征组合')

    parser.add_argument('--target', type=str, default='toxic',
                        help='目标任务名称')

    parser.add_argument('--lr_scheduler', type=str, default='plateau',
                        choices=['step', 'cosine', 'plateau', 'none'],
                        help='学习率调度器类型')

    parser.add_argument('--list_features', action='store_true',
                        help='列出所有可用的特征组合并退出')

    args = parser.parse_args()

    # 如果请求列出特征
    if args.list_features:
        features_dict, _ = load_all_features()
        feature_combinations = generate_feature_combinations(features_dict)
        print("\n可用的特征组合:")
        for i, feature_name in enumerate(sorted(feature_combinations.keys()), 1):
            print(f"{i}. {feature_name}")
        return

    # 执行训练
    print(f"\n开始训练 - 模型: {args.model}, 特征: {args.feature}, 目标: {args.target}")

    results = train_model(
        feature_type=args.feature,
        model_type=args.model,
        target_name=args.target,
        lr_scheduler_type=args.lr_scheduler if args.lr_scheduler != 'none' else None
    )

    # 显示最终结果
    print("\n========== 训练完成 ==========")
    if results["best_fold"]:
        print(f"最佳模型来自第 {results['best_fold']['fold']} 折，ROC: {results['best_fold']['best_roc']:.4f}")
        print(f"最终模型已保存到: {results['model_path']}")

        metrics = results["avg_metrics"]
        print(f"\n平均性能指标:")
        print(f"准确率: {metrics['acc']:.4f}")
        print(f"灵敏度: {metrics['sn']:.4f}")
        print(f"特异性: {metrics['sp']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"ROC曲线下面积: {metrics['roc']:.4f}")
        print(f"PR曲线下面积: {metrics['pr_auc']:.4f}")
    else:
        print("训练未产生有效模型。请检查训练参数和数据。")


if __name__ == '__main__':
    main()
