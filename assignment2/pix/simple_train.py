import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from enhanced_FCN_network import EnhancedFullyConvNetwork
from discriminator import PatchGANDiscriminator

def tensor_to_image(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, (1, 2, 0))
    image = (image + 1) / 2
    image = (image * 255).astype(np.uint8)
    
    # 【新增】将 RGB 转回 BGR，以适配 cv2.imwrite
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image

def save_sample_images(inputs, targets, outputs, folder_name, iteration, num_images=3):
    """
    保存输入、目标和输出图像的示例
    
    Args:
        inputs (torch.Tensor): 输入图像批次
        targets (torch.Tensor): 目标图像批次
        outputs (torch.Tensor): 输出图像批次
        folder_name (str): 保存图像的目录
        iteration (int): 当前迭代次数
        num_images (int): 要保存的图像数量
    """
    os.makedirs(folder_name, exist_ok=True)
    
    # 确定要保存的图像数量
    actual_num = min(num_images, inputs.size(0))
    
    for i in range(actual_num):
        # 将张量转换为图像
        input_img = tensor_to_image(inputs[i])
        target_img = tensor_to_image(targets[i])
        output_img = tensor_to_image(outputs[i])

        # 水平拼接图像
        comparison = np.hstack((input_img, target_img, output_img))

        # 保存拼接图像
        cv2.imwrite(f'{folder_name}/sample_iter_{iteration}_img_{i+1}.png', comparison)


def main():
    """
    主函数，设置并执行训练过程
    """
    # 获取当前脚本的绝对目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接列表文件的绝对路径
    train_list_path = os.path.join(current_dir, 'train_list.txt')
    val_list_path = os.path.join(current_dir, 'val_list.txt')

    # 检查文件是否存在
    if not os.path.exists(train_list_path):
        raise FileNotFoundError(f"训练集列表文件不存在: {train_list_path}\n请先运行 zhuanhuan.py 生成该文件。")
    if not os.path.exists(val_list_path):
        raise FileNotFoundError(f"验证集列表文件不存在: {val_list_path}\n请先运行 zhuanhuan.py 生成该文件。")

    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据集和数据加载器
    train_dataset = FacadesDataset(list_file=train_list_path)
    val_dataset = FacadesDataset(list_file=val_list_path)

    # 使用较小的batch size以适应显存限制
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化生成器和判别器
    generator = EnhancedFullyConvNetwork().to(device)
    discriminator = PatchGANDiscriminator(input_nc=3, ndf=64, n_layers=3).to(device)

    # 定义损失函数
    criterion_GAN = nn.MSELoss()
    criterion_L1 = nn.L1Loss()
    
    # 定义优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 训练参数
    num_epochs = 300
    n_epochs_decay = 150  # 开始学习率衰减的epoch数
    lambda_L1 = 100  # L1损失的权重，Pix2Pix论文推荐值
    save_interval = 100
    val_interval = 10
    
    # 标签平滑参数（防止判别器过强）
    real_label_smooth = 0.9

    # 训练循环
    global_step = 0
    best_val_loss = float('inf')
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Learning rate decay starts at epoch {n_epochs_decay}")
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        running_loss_G = 0.0
        running_loss_D = 0.0
        
        # 学习率衰减策略（Pix2Pix论文推荐）
        if epoch > n_epochs_decay:
            lr_decay_factor = 1.0 - (epoch - n_epochs_decay) / (num_epochs - n_epochs_decay)
            current_lr = 0.0002 * lr_decay_factor
            for param_group in optimizer_G.param_groups:
                param_group['lr'] = current_lr
            for param_group in optimizer_D.param_groups:
                param_group['lr'] = current_lr
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 移动数据到设备
            data, target = data.to(device), target.to(device)
            
            batch_size = data.size(0)
            
            # ========================
            # 训练判别器
            # ========================
            optimizer_D.zero_grad()
            
            # 真实图像的判别结果
            pred_real = discriminator(data, target)
            real_labels = torch.full_like(pred_real, real_label_smooth, requires_grad=False)
            loss_D_real = criterion_GAN(pred_real, real_labels)
            
            # 生成图像的判别结果
            fake_output = generator(data)
            pred_fake = discriminator(data, fake_output.detach())
            fake_labels = torch.zeros_like(pred_fake, requires_grad=False)
            loss_D_fake = criterion_GAN(pred_fake, fake_labels)
            
            # 判别器总损失
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ========================
            # 训练生成器
            # ========================
            optimizer_G.zero_grad()
            
            # GAN损失：希望生成的图像被判别为真实
            pred_fake_for_G = discriminator(data, fake_output)
            loss_G_GAN = criterion_GAN(pred_fake_for_G, real_labels)
            
            # L1损失：生成图像与目标图像的像素级差异
            loss_G_L1 = criterion_L1(fake_output, target)
            
            # 生成器总损失
            loss_G = loss_G_GAN + lambda_L1 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()
            
            running_loss_G += loss_G.item()
            running_loss_D += loss_D.item()
            
            # 每隔一定步数保存样本图像
            if global_step % save_interval == 0:
                save_sample_images(data, target, fake_output.detach(), 'samples', global_step)
                print(f'Iteration [{global_step}], Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}, Loss_GAN: {loss_G_GAN.item():.4f}, Loss_L1: {loss_G_L1.item():.4f}')
                
            global_step += 1

        # 打印每个epoch的平均损失
        avg_loss_G = running_loss_G / len(train_loader)
        avg_loss_D = running_loss_D / len(train_loader)
        current_lr = optimizer_G.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Avg Loss_G: {avg_loss_G:.4f}, Avg Loss_D: {avg_loss_D:.4f}, LR: {current_lr:.6f}')
        
        # 验证阶段
        if (epoch + 1) % val_interval == 0:
            generator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_data, val_target in val_loader:
                    val_data, val_target = val_data.to(device), val_target.to(device)
                    val_output = generator(val_data)
                    val_loss += criterion_L1(val_output, val_target).item()
            
            avg_val_loss = val_loss / len(val_loader)
            print(f'>>> Validation Loss: {avg_val_loss:.4f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': avg_val_loss,
                }, f'models/pix2pix_best.pth')
                print(f'>>> Best model saved with val_loss: {avg_val_loss:.4f}')
        
        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'epoch': epoch + 1,
            }, f'models/pix2pix_epoch_{epoch+1}.pth')
            print(f'Model checkpoint saved at epoch {epoch+1}')

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()