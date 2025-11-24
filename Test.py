import csv
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import (recall_score, roc_auc_score, roc_curve, precision_recall_curve, auc)
import pandas as pd
from util.Train_util import load_data_pkl, get_CM
from models.Model import OptimizedCNN_Model, CNN_Model, CustomDataset
import os
from MLP_train import MLP
import joblib
import argparse
from data_loader.data_loader import load_all_features, generate_feature_combinations
from util.model_utils import setup_paths
import shap
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, output_dir, model_type, feature_type):
    """
    绘制并保存混淆矩阵

    Args:
        y_true: 真实标签
        y_pred: 预测标签
        output_dir: 输出目录
        model_type: 模型类型
        feature_type: 特征类型
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 创建图形
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])

    plt.title(f'Confusion Matrix - {model_type} with {feature_type}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_type}_{feature_type}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"混淆矩阵已保存到 {cm_path}")

    # 返回混淆矩阵数据
    return cm


def extract_and_visualize_features(model, dataloader, device, model_type, output_dir, feature_type, labels=None):
    """
    提取模型最后一层特征并使用t-SNE可视化

    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 计算设备(CPU/GPU)
        model_type: 模型类型
        output_dir: 输出目录
        feature_type: 特征类型
        labels: 数据标签
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import os
    import numpy as np

    model.eval()
    features_list = []
    labels_list = []

    # 注册一个钩子来捕获倒数第二层的输出
    features = {}

    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach().cpu()

        return hook

    # 根据不同的模型类型，注册不同的钩子
    if model_type == "OptimizedCNN":
        # 为倒数第二层注册钩子 (根据您的模型架构可能需要调整)
        hook_handle = model.mlp[0].register_forward_hook(get_activation('features'))
    elif model_type == "MLP":
        hook_handle = model.layers[-2].register_forward_hook(get_activation('features'))
    else:
        # 默认钩子，可能需要根据实际模型架构调整
        try:
            hook_handle = list(model.children())[-2].register_forward_hook(get_activation('features'))
        except:
            print(f"无法为{model_type}模型注册钩子，将使用备用方法")
            # 备用方法，可能适用于某些模型
            if hasattr(model, 'fc'):
                hook_handle = model.fc.register_forward_hook(get_activation('features'))
            else:
                print(f"警告: 无法为{model_type}模型自动添加特征提取钩子，请手动检查模型结构并修改代码")
                return

    print("开始提取特征...")
    # 提取特征
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            # 修改建议
            # 添加到test.py中
            if model_type in ["CNN", "OptimizedCNN"]:
                data = data.unsqueeze(1)  # 只有2D特征需要额外维度

            elif model_type == "MLP":
                data = data.view(data.size(0), -1)  # MLP需要展平输入

            # 前向传播
            _ = model(data)

            # 保存这批次的特征和标签
            features_batch = features['features'].cpu().numpy()
            features_list.append(features_batch)
            labels_list.append(label.cpu().numpy())

    # 移除钩子
    hook_handle.remove()

    # 合并所有批次的特征和标签
    all_features = np.vstack(features_list)
    all_labels = np.concatenate(labels_list).ravel()

    print(f"提取了 {all_features.shape[0]} 个样本的特征，特征维度: {all_features.shape[1]}")

    # 应用t-SNE降维
    print("正在应用t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(all_features)

    # 创建可视化图
    plt.figure(figsize=(10, 8))

    # 获取正负样本的索引
    pos_indices = all_labels == 1
    neg_indices = all_labels == 0

    # scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=all_labels, cmap='viridis', alpha=0.7)

    # 分别为正负样本绘制散点图，设置不同的颜色和较高的透明度
    plt.scatter(tsne_results[pos_indices, 0], tsne_results[pos_indices, 1],
                c='red', alpha=0.3, label='Positive')
    plt.scatter(tsne_results[neg_indices, 0], tsne_results[neg_indices, 1],
                c='blue', alpha=0.3, label='Negative')

    # 添加图例，设置在右上角
    plt.legend(loc='upper right')

    plt.title(f't-SNE Visualization of {model_type} Features for {feature_type}')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')

    # 保存图像
    os.makedirs(output_dir, exist_ok=True)
    tsne_plot_path = os.path.join(output_dir, f'tsne_{model_type}_{feature_type}.png')
    plt.savefig(tsne_plot_path)
    plt.close()
    print(f"t-SNE可视化已保存到 {tsne_plot_path}")

    return tsne_results, all_labels

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            output = outputs[0]  # 如果是元组，取第一个元素
        else:
            output = outputs
        if output.dim() == 1:
            output = output.unsqueeze(1)  # 如果是一维，转换为二维 (batch_size, 1)
        return output

def compute_shap_with_kernel_explainer(model, dataloader, device, model_type, output_dir, feature_type, num_samples=200):
    model.eval()

    # 提取背景数据和测试数据
    background_data = []
    test_data = []
    for data, _ in dataloader:
        data = data.to(device)
        if model_type in ["CNN", "OptimizedCNN"]:
            data = data.view(data.size(0), -1)
        background_data.append(data.cpu().numpy())
        test_data.append(data.cpu().numpy())

    # 使用全部 2208 个样本作为背景数据
    background_data_np = np.vstack(background_data)[:500]  # 形状: (2208, 2304)
    test_data_np = np.vstack(test_data)[:num_samples]       # 形状: (2208, 2304)

    print(f"背景数据形状: {background_data_np.shape}")
    print(f"测试数据形状: {test_data_np.shape}")

    # 定义预测函数
    def predict_function(x):
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        if model_type in ["CNN", "OptimizedCNN"]:
            x_tensor = x_tensor.view(x_tensor.size(0), 1, -1)
        with torch.no_grad():
            outputs = model(x_tensor)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            if outputs.dim() > 1:
                outputs = outputs[:, 0]
        return outputs.cpu().numpy()

    # 估计噪声方差
    background_preds = predict_function(background_data_np)
    noise_variance = np.var(background_preds)  # 使用预测值方差作为粗略估计
    print(f"估计的噪声方差: {noise_variance}")

    # 计算 SHAP 值
    explainer = shap.KernelExplainer(predict_function, background_data_np)
    shap_values = explainer.shap_values(
        test_data_np,
        nsamples=500,
        l1_reg="num_features(20)",  # 限制选择 20 个特征,
    ridge = 0.1  # 增加岭正则化参数，提高数值稳定性
    )

    # 绘制 SHAP 条形图
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, test_data_np, plot_type="bar", max_display=20, show=False)
    shap_bar_path = os.path.join(output_dir, f'shap_bar_{model_type}_{feature_type}.png')
    plt.savefig(shap_bar_path)
    plt.close()
    print(f"SHAP 条形图已保存到 {shap_bar_path}")

    # 绘制 SHAP 点图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, test_data_np, max_display=20, show=False)
    shap_dot_path = os.path.join(output_dir, f'shap_dot_{model_type}_{feature_type}.png')
    plt.savefig(shap_dot_path)
    plt.close()
    print(f"SHAP 点图已保存到 {shap_dot_path}")

    # 保存SHAP值以便后续分析
    np.save(os.path.join(output_dir, f'shap_values_{model_type}_{feature_type}.npy'), shap_values)

def main(model_type, feature_type, target_name="toxic", lr_scheduler_type='plateau'):

    paths = setup_paths(model_type, feature_type, target_name)

    y2, t5 = load_data_pkl(
        './data_loader/Prott5/test_pos.pkl',
        './data_loader/Prott5/test_neg.pkl')

    y3, esm = load_data_pkl(
        './data_loader/esm1b_t33_650M_UR50S/test_pos.pkl',
        './data_loader/esm1b_t33_650M_UR50S/test_neg.pkl')


    print(f"开始推理模型 - 特征类型: {feature_type}, 模型类型: {model_type}")

    # 加载所有特征数据
    features_dict, labels_dict = load_all_features()


    # 使用自动设置的路径
    scaler_path = paths["scaler"]
    test_roc_path = paths["test_roc"]
    test_pr_path = paths["test_pr"]
    model_path = paths["model"]
    csv_file = paths["test_results"]
    output_dir = paths["tsne_output"]

    feature_combinations = generate_feature_combinations(features_dict)
    selected_raw_feature = feature_combinations[feature_type]

    print(f"检测到2D特征，形状: {selected_raw_feature.shape}")
    input_channel = selected_raw_feature.shape[1]  # 特征维度

    print(f"已选择特征 '{feature_type}'，形状: {selected_raw_feature.shape}")
    print(f"动态计算得到的 input_channel: {input_channel}")

    try:
        # 尝试加载标准化器
        scaler = joblib.load(scaler_path)
        print(f"成功加载标准化器: {scaler_path}")
        # 应用标准化
        test_feature_scaled = scaler.transform(selected_raw_feature)
        print("使用标准化特征进行预测")
    except (FileNotFoundError, IOError) as e:
        # 标准化器文件不存在或无法加载
        print(f"警告: 无法加载标准化器 ({e})")
        print("跳过标准化步骤，直接使用原始特征")
        # 使用原始特征
        test_feature_scaled =selected_raw_feature

    print(f't5.shape:{t5.shape}')
    print(f'y2.shape:{y2.shape}')
    print(f'y3.shape:{y3.shape}')
    print(f'esm.shape:{esm.shape}')

    # 创建测试数据集
    test_dataset = CustomDataset(test_feature_scaled, labels_dict["t5"])
    test_loader = DataLoader(test_dataset, batch_size=16)

    models = {
        "CNN": CNN_Model,
        "OptimizedCNN": OptimizedCNN_Model,
        "MLP": MLP  # 添加MLP到模型字典
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device:{device}')

    # 新增：定义 OptimizedCNN 的超参数
    hidden_dim = 512        # 第一个卷积层的输出通道数
    dropout_rate = 0.4      # Dropout 比例
    activation = 'gelu'     # 激活函数类型

    # 根据model_type实例化模型
    if model_type == "MLP":
        model = models[model_type](input_channel).to(device)  # MLP使用input_channel作为input_features
    elif model_type == "OptimizedCNN":
        model = OptimizedCNN_Model(
                    input_channels=input_channel,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    activation=activation
                ).to(device)
    else:
        model = models[model_type](input_channel).to(device)

    # 需要修改为
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=0.00001)

    # === 新增：定义与训练时一致的调度器 ===
    if lr_scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10, factor=0.5)
    else:
        scheduler = None

    # 4. 现在加载模型和优化器状态
    checkpoint = torch.load(model_path)
    try:
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # 加载完整的checkpoint字典
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("成功加载checkpoint格式的模型")
        else:
            # 直接加载state_dict
            model.load_state_dict(checkpoint)
            print("成功加载state_dict格式的模型")
    except Exception as e:
        print(f"模型加载出错: {e}")

    model.eval()
    print(f"初始学习率: {optimizer.param_groups[0]['lr']:.2e}")
    model.to(device)

    all_predictions = []
    all_labels = []
    all_auc = []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            labels = labels.squeeze(dim=-1)

            # 根据模型类型调整数据维度
            if model_type in ["CNN", "OptimizedCNN"]:
                data = data.unsqueeze(1)  # 只有2D特征需要额外维度
            elif model_type == "MLP":
                data = data.view(data.size(0), -1)  # MLP需要展平输入

            final_output = model(data)
            if isinstance(final_output, tuple):
                final_output = final_output[0]  # 获取第一个元素
            # if model_type != "MLP":  # MLP已在forward中squeeze，其他模型需要手动squeeze
            #     final_output = final_output.squeeze(1)
            scores = final_output.tolist()
            all_auc.extend(scores)
            final_output = (final_output.data > 0.5).int()
            all_labels.extend(labels.tolist())
            all_predictions.extend(final_output.tolist())

        acc, sn, sp, mcc, pr, f1 = get_CM(all_labels, all_predictions)
        # 绘制并保存混淆矩阵
        cm = plot_confusion_matrix(
            all_labels,
            all_predictions,
            output_dir,
            model_type,
            feature_type
        )
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


        pd.DataFrame({'FPR': fpr, 'TPR': tpr}).to_csv(test_roc_path, index=False)
        pd.DataFrame({'Precision': precision, 'Recall': recall}).to_csv(test_pr_path, index=False)

        print(
            f"acc:{test_accuracy:.4f} pr:{test_precision:.4f} recall{test_recall:.4f} "
            f"f1:{test_f1:.4f} roc:{test_auc_roc:.4f} auc:{test_pr_auc:.4f} "
            f"mcc:{mcc:.4f}sn:{sn:.4f}sp:{sp:.4f}")

        test_roc_curve_data.to_csv(test_roc_path, index=False)
        test_pr_curve_data.to_csv(test_pr_path, index=False)

        # 在这里调用t-SNE特征可视化函数
        # output_dir = './position/OptimizedCNN_dr_test_results/tsne_plots'
        extract_and_visualize_features(
            model=model,
            dataloader=test_loader,
            device=device,
            model_type=model_type,
            output_dir=output_dir,
            feature_type=feature_type,
            labels=labels_dict["t5"],
        )

        # 添加SHAP计算和可视化
        # 在t-SNE可视化之后
        if not args.no_shap:
            # 添加SHAP计算和可视化
            compute_shap_with_kernel_explainer(
                model=model,
                dataloader=test_loader,
                device=device,
                model_type=model_type,
                output_dir=output_dir,
                feature_type=feature_type,
                num_samples=2208  # 可根据需要调整
            )
        else:
            print("跳过SHAP值计算")

        '''save acc'''

        file_exists = os.path.exists(csv_file)
        with open(csv_file, 'a', newline='') as file1:
            writer = csv.writer(file1, dialect='excel')
            # 如果文件不存在，则先写入表头
            if not file_exists:
                writer.writerow(["Tag", "Accuracy", "Sensitivity", "Specificity", "MCC", "F1", "Recall", "ROC_AUC", "PR_AUC"])
            label = f'{target_name}: {feature_type} test'

            writer.writerow([label, acc, sn, sp, mcc, f1, test_recall, test_auc_roc, test_pr_auc, f"{optimizer.param_groups[0]['lr']:.2e}"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='测试蛋白质特征分析模型')

    # 添加参数
    parser.add_argument('--model', type=str, default='OptimizedCNN',
                        choices=['CNN', 'OptimizedCNN', 'MLP'],
                        help='要使用的模型类型')

    parser.add_argument('--feature', type=str, default='t5_esm',
                        help='要使用的特征类型或特征组合')

    parser.add_argument('--target', type=str, default='toxic',
                        help='目标任务名称')

    parser.add_argument('--no_shap', action='store_true',
                        help='如果设置此参数，则不计算SHAP值')

    args = parser.parse_args()



    # 执行测试
    print(f"\n开始测试 - 模型: {args.model}, 特征: {args.feature}, 目标: {args.target}")
    main(args.model, args.feature, args.target)

