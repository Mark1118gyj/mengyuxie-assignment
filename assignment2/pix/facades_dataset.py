import torch
from torch.utils.data import Dataset
import cv2
import os

class FacadesDataset(Dataset):
    def __init__(self, list_file, base_dir=None):
        """
        Args:
            list_file (string): Path to the txt file with image filenames.
            base_dir (string): Base directory for resolving relative paths. 
                              If None, uses the directory containing list_file.
        """
        # 获取list_file所在的目录
        list_file_dir = os.path.dirname(os.path.abspath(list_file))
        
        # Read the list of image filenames
        with open(list_file, 'r') as file:
            all_filenames = [line.strip() for line in file if line.strip()]
        
        # 如果没有指定base_dir，使用list_file所在目录作为基准
        if base_dir is None:
            self.base_dir = list_file_dir
        else:
            self.base_dir = base_dir
        
        print(f"Dataset configuration:")
        print(f"  List file: {list_file}")
        print(f"  Base directory: {self.base_dir}")
        print(f"  Current working directory: {os.getcwd()}")
        print()
        
        # 验证并过滤存在的文件
        self.image_filenames = []
        missing_count = 0
        sample_paths_shown = 0
        
        for img_path in all_filenames:
            # 处理各种路径格式
            if os.path.isabs(img_path):
                # 绝对路径，直接使用
                full_path = img_path
            elif img_path.startswith('./') or img_path.startswith('../'):
                # 相对路径（以./或../开头）
                # 先尝试相对于base_dir
                full_path = os.path.join(self.base_dir, img_path)
                
                # 如果不存在，尝试相对于当前工作目录
                if not os.path.exists(full_path):
                    alt_path = os.path.abspath(img_path)
                    if os.path.exists(alt_path):
                        full_path = alt_path
            else:
                # 普通相对路径
                full_path = os.path.join(self.base_dir, img_path)
            
            # 规范化路径
            full_path = os.path.normpath(full_path)
            
            # 检查文件是否存在
            if os.path.exists(full_path):
                self.image_filenames.append(img_path)
                
                # 显示前几个成功找到的文件用于调试
                if sample_paths_shown < 3:
                    print(f"  ✓ Found: {img_path}")
                    print(f"    Resolved to: {full_path}")
                    sample_paths_shown += 1
            else:
                missing_count += 1
                # 显示前几个缺失的文件用于调试
                if missing_count <= 5:
                    print(f"  ✗ Missing: {img_path}")
                    print(f"    Tried: {full_path}")
        
        print()
        
        if missing_count > 0:
            print(f"Warning: {missing_count} files not found and will be skipped.")
        
        if len(self.image_filenames) == 0:
            # 提供详细的调试信息
            print(f"\n{'='*70}")
            print(f"ERROR: No valid images found!")
            print(f"{'='*70}")
            print(f"List file: {list_file}")
            print(f"Base directory: {self.base_dir}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"\nFirst few entries from list file:")
            for path in all_filenames[:5]:
                if os.path.isabs(path):
                    test_path = path
                elif path.startswith('./') or path.startswith('../'):
                    test_path = os.path.normpath(os.path.join(self.base_dir, path))
                else:
                    test_path = os.path.normpath(os.path.join(self.base_dir, path))
                
                exists = "✓ EXISTS" if os.path.exists(test_path) else "✗ MISSING"
                print(f"  {path}")
                print(f"    -> {test_path} [{exists}]")
            
            # 检查datasets目录是否存在
            datasets_path = os.path.join(self.base_dir, 'datasets')
            if os.path.exists(datasets_path):
                print(f"\n✓ datasets directory exists at: {datasets_path}")
                facades_path = os.path.join(datasets_path, 'facades')
                if os.path.exists(facades_path):
                    print(f"✓ facades directory exists at: {facades_path}")
                    train_path = os.path.join(facades_path, 'train')
                    if os.path.exists(train_path):
                        files = os.listdir(train_path)
                        print(f"✓ train directory exists with {len(files)} files")
                        if files:
                            print(f"  First file: {files[0]}")
                    else:
                        print(f"✗ train directory NOT found at: {train_path}")
                else:
                    print(f"✗ facades directory NOT found")
            else:
                print(f"\n✗ datasets directory NOT found at: {datasets_path}")
                print(f"   Please run: bash download_facades_dataset.sh")
            
            print(f"{'='*70}\n")
            raise FileNotFoundError(f"No valid images found in {list_file}")
        
        print(f"Successfully loaded {len(self.image_filenames)} valid images\n")
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename
        img_name = self.image_filenames[idx]
        
        # 构建完整路径（与__init__中相同的逻辑）
        if os.path.isabs(img_name):
            img_path = img_name
        elif img_name.startswith('./') or img_name.startswith('../'):
            img_path = os.path.join(self.base_dir, img_name)
            if not os.path.exists(img_path):
                alt_path = os.path.abspath(img_name)
                if os.path.exists(alt_path):
                    img_path = alt_path
        else:
            img_path = os.path.join(self.base_dir, img_name)
        
        # 规范化路径
        img_path = os.path.normpath(img_path)
        
        # 读取图像
        img_color_semantic = cv2.imread(img_path)
        
        # 检查是否成功读取
        if img_color_semantic is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}\nFile may be corrupted or inaccessible.")
        
        # 检查图像尺寸是否正确（应该是 256x512）
        height, width = img_color_semantic.shape[:2]
        if width != 512 or height != 256:
            print(f"Warning: Image {img_path} has unexpected size {width}x{height}, resizing to 512x256")
            img_color_semantic = cv2.resize(img_color_semantic, (512, 256))
        
        # Convert the image to a PyTorch tensor
        # BGR to RGB conversion is needed because OpenCV loads in BGR format
        img_rgb = cv2.cvtColor(img_color_semantic, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
        
        # 分割为RGB和语义图（各256宽度）
        image_real = image[:, :, :256]      # 左边256列 - 真实图
        image_semantic = image[:, :, 256:]  # 右边256列 - 语义图
        
        return image_semantic, image_real