import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from enhanced_FCN_network import EnhancedFullyConvNetwork
import matplotlib.pyplot as plt
from tqdm import tqdm

def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    
    # 【新增】将 RGB 转回 BGR，以适配 cv2.imwrite
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    image = np.ascontiguousarray(image)
    return image

def save_test_results(inputs, targets, outputs, save_dir, indices=None):
    """
    保存测试结果的对比图
    
    Args:
        inputs: 输入图像批次
        targets: 目标图像批次
        outputs: 生成图像批次
        save_dir: 保存目录
        indices: 要保存的图像索引列表，None表示保存所有
    """
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = inputs.size(0)
    if indices is None:
        indices = range(batch_size)
    
    for i in indices:
        if i >= batch_size:
            break
            
        try:
            input_img = tensor_to_image(inputs[i])
            target_img = tensor_to_image(targets[i])
            output_img = tensor_to_image(outputs[i])

            # 确保数组是C连续的（OpenCV要求）
            input_img = np.ascontiguousarray(input_img)
            target_img = np.ascontiguousarray(target_img)
            output_img = np.ascontiguousarray(output_img)

            # 添加标签（在拼接之前）
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            color = (255, 255, 255)
            
            cv2.putText(input_img, 'Input', (10, 25), font, font_scale, color, thickness)
            cv2.putText(target_img, 'Ground Truth', (10, 25), font, font_scale, color, thickness)
            cv2.putText(output_img, 'Generated', (10, 25), font, font_scale, color, thickness)
            
            # 水平拼接：输入 | 真实目标 | 生成结果
            comparison = np.hstack((input_img, target_img, output_img))

            # 保存拼接图
            output_path = f'{save_dir}/test_result_{i+1}.png'
            success = cv2.imwrite(output_path, comparison)
            if not success:
                print(f"Warning: Failed to save {output_path}")
            
            # 保存单独的生成结果
            gen_path = f'{save_dir}/generated_{i+1}.png'
            cv2.imwrite(gen_path, output_img)
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
            continue

def calculate_metrics(outputs, targets):
    """
    计算评估指标
    
    Args:
        outputs: 生成的图像张量
        targets: 真实的目标图像张量
    
    Returns:
        dict: 包含各种评估指标的字典
    """
    # 确保在CPU上计算
    outputs_cpu = outputs.cpu().detach()
    targets_cpu = targets.cpu().detach()
    
    # L1距离（MAE）
    l1_loss = torch.mean(torch.abs(outputs_cpu - targets_cpu)).item()
    
    # L2距离（MSE）
    l2_loss = torch.mean((outputs_cpu - targets_cpu) ** 2).item()
    
    # PSNR (Peak Signal-to-Noise Ratio)
    mse = l2_loss
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(2.0 / np.sqrt(mse))  # 数据范围是[-1, 1]，所以最大值差是2
    
    # SSIM (Structural Similarity Index) - 简化版本
    # 将数据从[-1, 1]转换到[0, 1]
    outputs_norm = (outputs_cpu + 1) / 2
    targets_norm = (targets_cpu + 1) / 2
    
    # 计算均值
    mu_pred = torch.mean(outputs_norm, dim=[2, 3], keepdim=True)
    mu_true = torch.mean(targets_norm, dim=[2, 3], keepdim=True)
    
    # 计算方差
    sigma_pred = torch.var(outputs_norm, dim=[2, 3], keepdim=True, unbiased=False)
    sigma_true = torch.var(targets_norm, dim=[2, 3], keepdim=True, unbiased=False)
    
    # 计算协方差
    sigma_pred_true = torch.mean((outputs_norm - mu_pred) * (targets_norm - mu_true), dim=[2, 3], keepdim=True)
    
    # SSIM参数
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # 计算SSIM
    ssim_num = (2 * mu_pred * mu_true + C1) * (2 * sigma_pred_true + C2)
    ssim_den = (mu_pred ** 2 + mu_true ** 2 + C1) * (sigma_pred + sigma_true + C2)
    ssim = torch.mean(ssim_num / ssim_den).item()
    
    return {
        'L1': l1_loss,
        'MSE': l2_loss,
        'PSNR': psnr,
        'SSIM': ssim
    }

def test_model(model_path, test_list_path, save_dir='test_results', num_samples=10, batch_size=4):
    """
    在测试集上测试模型
    
    Args:
        model_path: 模型文件路径
        test_list_path: 测试集列表文件路径
        save_dir: 结果保存目录
        num_samples: 保存的样本数量
        batch_size: 批大小
    """
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # 检查测试集列表文件
    if not os.path.exists(test_list_path):
        raise FileNotFoundError(f"Test list file not found: {test_list_path}")
    
    print(f"Loading model from: {model_path}")
    print(f"Test list: {test_list_path}")
    
    # 创建测试数据集
    test_dataset = FacadesDataset(list_file=test_list_path)
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化生成器
    generator = EnhancedFullyConvNetwork().to(device)
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    
    # 尝试不同的键名
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_loss' in checkpoint:
            print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    else:
        generator.load_state_dict(checkpoint)
        print("Loaded model weights directly")
    
    generator.eval()
    print("Model loaded successfully!\n")
    
    # 测试循环
    all_metrics = []
    sample_count = 0
    saved_indices = []
    
    print("Testing on test set...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # 生成结果
            output = generator(data)
            
            # 计算指标
            metrics = calculate_metrics(output, target)
            all_metrics.append(metrics)
            
            # 保存前num_samples个样本
            if sample_count < num_samples:
                samples_to_save = min(num_samples - sample_count, data.size(0))
                
                # 记录哪些索引被保存了
                for i in range(samples_to_save):
                    global_idx = sample_count + i
                    saved_indices.append(global_idx)
                
                save_test_results(
                    data, target, output,
                    f'{save_dir}/samples',
                    indices=range(samples_to_save)
                )
                
                sample_count += samples_to_save
            
            # 更新进度条
            pbar.set_postfix({
                'L1': f"{metrics['L1']:.4f}",
                'PSNR': f"{metrics['PSNR']:.2f}"
            })
    
    # 计算平均指标
    avg_metrics = {
        key: np.mean([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    std_metrics = {
        key: np.std([m[key] for m in all_metrics])
        for key in all_metrics[0].keys()
    }
    
    # 打印结果
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    print(f"Total test samples: {len(test_dataset)}")
    print(f"Samples saved: {min(num_samples, len(test_dataset))}")
    print(f"Results saved to: {save_dir}/")
    print("-"*70)
    print(f"{'Metric':<15} {'Mean':<15} {'Std':<15}")
    print("-"*70)
    for key in avg_metrics.keys():
        print(f"{key:<15} {avg_metrics[key]:<15.6f} {std_metrics[key]:<15.6f}")
    print("="*70)
    
    # 保存指标到文件
    metrics_file = os.path.join(save_dir, 'test_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Test Results Summary\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test set: {test_list_path}\n")
        f.write(f"Total samples: {len(test_dataset)}\n\n")
        
        f.write(f"{'Metric':<15} {'Mean':<15} {'Std':<15}\n")
        f.write("-"*70 + "\n")
        for key in avg_metrics.keys():
            f.write(f"{key:<15} {avg_metrics[key]:<15.6f} {std_metrics[key]:<15.6f}\n")
        f.write("="*70 + "\n")
    
    print(f"\nMetrics saved to: {metrics_file}")
    
    # 可视化一些结果
    visualize_results(save_dir, num_samples=min(5, len(saved_indices)))
    
    return avg_metrics

def visualize_results(save_dir, num_samples=5):
    """
    可视化测试结果
    """
    samples_dir = os.path.join(save_dir, 'samples')
    
    if not os.path.exists(samples_dir):
        print(f"No samples found in {samples_dir}")
        return
    
    # 获取保存的结果文件
    result_files = sorted([
        f for f in os.listdir(samples_dir) 
        if f.startswith('test_result_') and f.endswith('.png')
    ])
    
    if not result_files:
        print("No result files found")
        return
    
    num_to_show = min(num_samples, len(result_files))
    
    fig, axes = plt.subplots(num_to_show, 1, figsize=(15, 5*num_to_show))
    if num_to_show == 1:
        axes = [axes]
    
    for i in range(num_to_show):
        img_path = os.path.join(samples_dir, result_files[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Sample {i+1}', fontsize=14)
    
    plt.tight_layout()
    viz_path = os.path.join(save_dir, 'visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    plt.close()

def main():
    """
    主函数
    """
    # 获取当前脚本的绝对目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 模型路径 - 使用最佳模型
    model_path = os.path.join(current_dir, 'models', 'pix2pix_best.pth')
    
    # 如果没有最佳模型，使用最后一个checkpoint
    if not os.path.exists(model_path):
        # 查找最新的epoch模型
        models_dir = os.path.join(current_dir, 'models')
        if os.path.exists(models_dir):
            epoch_models = [
                f for f in os.listdir(models_dir) 
                if f.startswith('pix2pix_epoch_') and f.endswith('.pth')
            ]
            if epoch_models:
                # 按epoch编号排序
                epoch_models.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                model_path = os.path.join(models_dir, epoch_models[-1])
                print(f"Best model not found, using latest: {model_path}")
            else:
                raise FileNotFoundError("No trained models found in models/")
        else:
            raise FileNotFoundError("Models directory not found. Please train the model first.")
    
    # 测试集列表文件
    test_list_path = os.path.join(current_dir, 'test_list.txt')
    
    # 如果test_list.txt不存在，尝试使用val_list.txt
    if not os.path.exists(test_list_path):
        val_list_path = os.path.join(current_dir, 'val_list.txt')
        if os.path.exists(val_list_path):
            print("Warning: test_list.txt not found, using val_list.txt instead")
            test_list_path = val_list_path
        else:
            raise FileNotFoundError(
                "Test list file not found. Please create test_list.txt or ensure val_list.txt exists."
            )
    
    # 结果保存目录
    save_dir = os.path.join(current_dir, 'test_results')
    
    # 运行测试
    print("="*70)
    print("PIX2PIX MODEL TESTING")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Test set: {test_list_path}")
    print(f"Save directory: {save_dir}")
    print("="*70 + "\n")
    
    metrics = test_model(
        model_path=model_path,
        test_list_path=test_list_path,
        save_dir=save_dir,
        num_samples=10,
        batch_size=4
    )
    
    print("\nTesting completed!")

if __name__ == '__main__':
    main()