import numpy as np
import matplotlib.pyplot as plt

def is_converged(train_losses, val_losses, train_accs, val_accs, window=10, loss_threshold=0.01, acc_threshold=0.005):
    """更完善的收敛判断，同时考虑损失和准确率"""
    if len(train_losses) <= window:
        return "数据不足以判断收敛"
        
    # 损失变化
    train_loss_change = abs(train_losses[-window] - train_losses[-1])
    val_loss_change = abs(val_losses[-window] - val_losses[-1])
    
    # 准确率变化
    train_acc_change = abs(train_accs[-window] - train_accs[-1])
    val_acc_change = abs(val_accs[-window] - val_accs[-1])
    
    # 判断准确率是否稳定而损失仍在下降(学习噪声)
    learning_noise = (train_acc_change < acc_threshold and val_acc_change < acc_threshold) and \
                     (train_loss_change > loss_threshold)
                     
    # 正常收敛: 损失和准确率都稳定
    normal_convergence = (train_loss_change < loss_threshold and val_loss_change < loss_threshold) and \
                         (train_acc_change < acc_threshold and val_acc_change < acc_threshold)
                         
    if learning_noise:
        return "收敛但可能学习噪声"
    elif normal_convergence:
        return "正常收敛"
    else:
        return "未完全收敛"

def analyze_training_results(train_acc_path, val_acc_path, train_loss_path, val_loss_path):
    # 读取数据文件
    train_accuracies = np.load(train_acc_path)
    val_accuracies = np.load(val_acc_path)
    train_losses = np.load(train_loss_path)
    val_losses = np.load(val_loss_path)
    
    # 获取最终值
    final_train_acc = train_accuracies[-1]
    final_val_acc = val_accuracies[-1]
    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1]
    
    # 计算差异
    acc_diff = final_train_acc - final_val_acc
    loss_diff = final_val_loss - final_train_loss
    
    # 分析收敛情况
    convergence_status = is_converged(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 分析过拟合情况
    # 一般来说，如果训练准确率明显高于验证准确率(>5%)或训练损失明显低于验证损失，可能存在过拟合
    overfitting = (acc_diff > 0.05) or (loss_diff > 0.1)
    
    # 绘制图表
    epochs = range(1, len(train_accuracies) + 1)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_accuracies, 'b-', label='训练集准确率')
    plt.plot(epochs, val_accuracies, 'r-', label='验证集准确率')
    plt.title('训练集与验证集准确率')
    plt.xlabel('Epochs')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_losses, 'b-', label='训练集损失')
    plt.plot(epochs, val_losses, 'r-', label='验证集损失')
    plt.title('训练集与验证集损失')
    plt.xlabel('Epochs')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    
    # 计算准确率收敛性
    acc_convergence = "稳定" if abs(train_accuracies[-10] - train_accuracies[-1]) < 0.005 else "仍在变化"
    
    # 计算验证集最佳表现
    best_val_acc = np.max(val_accuracies)
    best_val_acc_epoch = np.argmax(val_accuracies) + 1
    
    # 计算早停位置
    optimal_early_stop = False
    if best_val_acc_epoch < len(epochs):
        optimal_early_stop = True
    
    return {
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'acc_diff': acc_diff,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'loss_diff': loss_diff,
        'convergence_status': convergence_status,
        'acc_convergence': acc_convergence,
        'overfitting': overfitting,
        'best_val_acc': best_val_acc,
        'best_val_acc_epoch': best_val_acc_epoch,
        'optimal_early_stop': optimal_early_stop
    }

def main():
    # 文件路径
    base_path = '/home/dl01/Public/Add/lixiang/Alg-MFDL-main/position/OptimizedCNN_t5_esm_results/toxic/'
    train_acc_path = base_path + 'best_fold_train_accuracies.npy'
    val_acc_path = base_path + 'best_fold_val_accuracies.npy'
    train_loss_path = base_path + 'best_fold_train_losses.npy'
    val_loss_path = base_path + 'best_fold_val_losses.npy'
    
    # 分析结果
    results = analyze_training_results(train_acc_path, val_acc_path, train_loss_path, val_loss_path)
    
    # 输出结果
    print("=== 模型训练分析结果 ===")
    print(f"训练集最终准确率: {results['final_train_acc']:.4f}")
    print(f"验证集最终准确率: {results['final_val_acc']:.4f}")
    print(f"准确率差异: {results['acc_diff']:.4f}")
    print(f"训练集最终损失: {results['final_train_loss']:.4f}")
    print(f"验证集最终损失: {results['final_val_loss']:.4f}")
    print(f"损失差异: {results['loss_diff']:.4f}")
    print(f"验证集最佳准确率: {results['best_val_acc']:.4f} (第{results['best_val_acc_epoch']}轮)")
    
    # 提供结论
    print("\n=== 分析结论 ===")
    print(f"收敛状态: {results['convergence_status']}")
    print(f"准确率收敛性: {results['acc_convergence']}")
    
    if results['overfitting']:
        print("存在过拟合迹象: 训练集表现明显优于验证集。")
    else:
        print("无明显过拟合: 训练集和验证集表现相近。")
    
    if results['optimal_early_stop']:
        print(f"验证集性能在第{results['best_val_acc_epoch']}轮达到最佳，之后表现下降，建议使用早停。")
    
    # 综合结论
    print("\n=== 综合评估 ===")
    if "正常收敛" in results['convergence_status'] and not results['overfitting']:
        print("模型训练良好: 已收敛且无明显过拟合。")
    elif "正常收敛" in results['convergence_status'] and results['overfitting']:
        print("模型训练结果一般: 已收敛但存在过拟合。建议使用正则化技术或增加数据。")
    elif "噪声" in results['convergence_status']:
        print("警告: 模型可能在学习数据噪声。准确率已稳定但损失仍在变化。")
    elif "未完全收敛" in results['convergence_status'] and not results['overfitting']:
        print("模型训练可继续: 未完全收敛但无过拟合。可以增加训练轮数。")
    else:
        print("模型训练需改进: 未收敛且存在过拟合。建议重新调整模型结构或超参数。")
    
    print(f"\n已保存训练曲线图表到 'training_results.png'")

if __name__ == "__main__":
    main()
