import os
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import euler_angles_to_matrix
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === 1. 加载数据 ===
DATA_DIR = "data"
points2d_npz = np.load(os.path.join(DATA_DIR, "points2d.npz"))
colors = np.load(os.path.join(DATA_DIR, "points3d_colors.npy"))  # (20000, 3)

# 提取观测数据：views × points × (x, y, vis)
view_keys = [f"view_{i:03d}" for i in range(50)]
obs_list = [points2d_npz[key] for key in view_keys]  # list of (20000, 3)
obs_tensor = torch.tensor(np.stack(obs_list, axis=0), dtype=torch.float32, device=device)  # (50, 20000, 3)
obs_2d = obs_tensor[:, :, :2]      # (V, N, 2)
obs_vis = obs_tensor[:, :, 2] > 0.5  # (V, N), bool

V, N = obs_2d.shape[:2]
image_width, image_height = 1024, 1024
cx, cy = image_width / 2, image_height / 2

# === 2. 初始化优化变量 ===
# 3D points: (N, 3) - 创建新的可训练参数
X = torch.randn(N, 3, device=device, requires_grad=True) * 0.1
X.retain_grad()

# 相机外参：每视角 Euler 角和平移
euler_angles = torch.zeros(V, 3, device=device, requires_grad=True)  # (V, 3)
T = torch.zeros(V, 3, device=device, requires_grad=True)
T.data[:, 2] = -2.5  # 初始化平移 Z 分量为 -2.5

# 焦距 f（标量）
f = torch.tensor(886.0, device=device, requires_grad=True)  # 使用标量而非单元素张量

# === 3. 投影函数 ===
def project_points(X, euler, T, f):
    """
    X: (N, 3)
    euler: (V, 3)
    T: (V, 3)
    f: scalar
    Returns: (V, N, 2) projected 2D points
    """
    R = euler_angles_to_matrix(euler, convention="XYZ")  # (V, 3, 3)
    # 将 3D 点变换到各相机坐标系: [Xc, Yc, Zc] = R @ X^T + T
    Xc = torch.einsum('vij,nj->vni', R, X) + T.unsqueeze(1)  # (V, N, 3)
    Zc = Xc[:, :, 2]  # (V, N)
    
    # 避免除零
    eps = 1e-8
    Zc = torch.where(Zc.abs() < eps, eps * torch.sign(Zc), Zc)
    
    u = -f * Xc[:, :, 0] / Zc + cx
    v =  f * Xc[:, :, 1] / Zc + cy
    return torch.stack([u, v], dim=-1)  # (V, N, 2)

# === 4. 优化设置 ===
# 在创建优化器前验证参数是否为叶节点
print("Verifying parameters are leaf nodes:")
print(f"X.is_leaf: {X.is_leaf}")
print(f"euler_angles.is_leaf: {euler_angles.is_leaf}")
print(f"T.is_leaf: {T.is_leaf}")
print(f"f.is_leaf: {f.is_leaf}")

# 检查哪些参数不是叶节点并重新创建
if not X.is_leaf:
    X = X.clone().detach().requires_grad_(True)

optimizer = torch.optim.Adam([X, euler_angles, T, f], lr=1e-2)
losses = []

num_epochs = 2000

print("Starting optimization...")
for epoch in range(num_epochs):
    optimizer.zero_grad()
    
    # 前向投影
    pred_2d = project_points(X, euler_angles, T, f)  # (V, N, 2)
    
    # 计算重投影误差（仅可见点）
    error = (pred_2d - obs_2d)  # (V, N, 2)
    squared_error = (error ** 2).sum(dim=-1)  # (V, N)
    
    # 只对可见点求 loss
    loss = squared_error[obs_vis].mean()
    
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}, f: {f.item():.2f}")

# === 5. 保存结果 ===
# 保存 loss 曲线
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Reprojection Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")
print("Saved loss_curve.png")

# 保存重建的 3D 点云为 OBJ（带颜色）
X_final = X.detach().cpu().numpy()  # (N, 3)
colors_final = colors  # (N, 3), already in [0,1]

obj_lines = []
for i in range(N):
    x, y, z = X_final[i]
    r, g, b = colors_final[i]
    obj_lines.append(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")

with open("reconstructed_points.obj", "w") as f:
    f.writelines(obj_lines)

print("Saved reconstructed_points.obj")
print("Bundle Adjustment completed!")