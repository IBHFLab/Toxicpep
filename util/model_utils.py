import torch
import os
from models.Model import OptimizedCNN_Model

# 获取当前脚本的绝对路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def setup_paths(model_type, feature_type, target_name="toxic"):
    """设置模型和结果的路径"""

    # 训练模型路径
    model_base_folder = os.path.join(os.path.join(BASE_DIR,'position/'),
                                     f"{model_type}_{feature_type}_results")
    model_folder = os.path.join(model_base_folder, target_name)
    model_path = os.path.join(model_folder, 'model.pt')
    print(model_path)
    # 测试结果路径
    test_base_folder = os.path.join(os.path.join(BASE_DIR,'position/'),
                                    f"{model_type}_{feature_type}_test_results")
    os.makedirs(test_base_folder, exist_ok=True)

    paths = {
        "model": model_path,
        "scaler": os.path.join(model_folder, 'scaler.pkl'),
        "test_roc": os.path.join(test_base_folder, 'test_roc.csv'),
        "test_pr": os.path.join(test_base_folder, 'test_pr.csv'),
        "test_results": os.path.join(test_base_folder, 'test_results.csv'),
        "tsne_output": os.path.join(test_base_folder, 'tsne_plots')
    }

    os.makedirs(paths["tsne_output"], exist_ok=True)

    return paths


def load_model(model_path, input_channels):
    """加载预训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OptimizedCNN_Model(input_channels=input_channels)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model.to(device), device

