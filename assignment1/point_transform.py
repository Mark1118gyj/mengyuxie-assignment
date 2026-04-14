import cv2
import numpy as np
import gradio as gr

# 全局变量：用于存储源控制点和目标控制点
points_src = []
points_dst = []
image = None

# 当上传新图片时，重置控制点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# 记录点击的点并在图像上可视化
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    if image is None:
        return None
        
    # 调试：打印事件传递的坐标
    print(f"点击坐标: {evt.index}")
    
    # 获取点击的坐标
    x, y = evt.index[0], evt.index[1]

    # 交替点击以分别记录源点和目标点
    # 如果源点和目标点数量相等，则下一个点击的是源点；否则是目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # 在图像上绘制点（蓝色：源点，红色：目标点）和箭头
    marked_image = image.copy()
    
    # 绘制源点 (蓝色)
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 5, (255, 0, 0), -1)
        
    # 绘制目标点 (红色)
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 5, (0, 0, 255), -1)

    # 绘制从源点指向目标点的绿色箭头
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 2)

    return marked_image

# 基于移动最小二乘法 (MLS) 的点引导图像变形
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    使用移动最小二乘法 (MLS) 对图像进行变形。
    
    参数:
        image: 输入图像 (numpy array)
        source_pts: 源控制点数组 (N, 2)
        target_pts: 目标控制点数组 (N, 2)
        alpha: 权重函数的指数，控制局部性 (默认 1.0)
        eps: 防止除以零的小量
    
    返回:
        变形后的图像
    """
    if source_pts is None or target_pts is None or len(source_pts) == 0:
        return np.array(image)

    # 确保点是浮点数类型
    source_pts = np.asarray(source_pts, dtype=np.float32)
    target_pts = np.asarray(target_pts, dtype=np.float32)
    
    h, w = image.shape[:2]
    
    # 创建目标图像的坐标网格 (v)
    # vy, vx 形状为 (H, W)
    vy, vx = np.mgrid[0:h, 0:w]
    # v 形状为 (H, W, 2)，每个元素包含该像素的 (x, y) 坐标
    v = np.stack([vx, vy], axis=-1).astype(np.float32)
    
    num_points = source_pts.shape[0]
    
    # --- 核心逻辑：计算逆映射 ---
    # cv2.remap 需要知道目标图像的每个像素对应源图像的哪个位置。
    # MLS 通常定义正向映射 f(p_i) = q_i。
    # 为了得到逆映射，我们交换源点和目标点的角色，计算 g(q_i) = p_i。
    # 然后在目标图像的网格上评估 g(v)，得到的结果即为源图像坐标。
    
    p_inv = target_pts  # 定义域控制点 (目标点)
    q_inv = source_pts  # 值域控制点 (源点)
    
    # 扩展维度以便广播: (1, 1, N, 2)
    p_expanded = p_inv[np.newaxis, np.newaxis, :, :]
    q_expanded = q_inv[np.newaxis, np.newaxis, :, :]
    v_expanded = v[:, :, np.newaxis, :] # (H, W, 1, 2)
    
    # 1. 计算权重 wi = 1 / |pi - v|^alpha
    diff = p_expanded - v_expanded # (H, W, N, 2)
    dist_sq = np.sum(diff**2, axis=-1) # (H, W, N)
    dist_sq = np.maximum(dist_sq, eps) # 避免除以零
    weights = 1.0 / (dist_sq ** alpha) # (H, W, N)
    
    # 2. 计算加权质心 p_hat 和 q_hat
    sum_w = np.sum(weights, axis=-1, keepdims=True) # (H, W, 1)
    p_hat = np.sum(weights[:, :, :, np.newaxis] * p_expanded, axis=-2) / sum_w # (H, W, 2)
    q_hat = np.sum(weights[:, :, :, np.newaxis] * q_expanded, axis=-2) / sum_w # (H, W, 2)
    
    # 3. 计算去中心化的点
    p_centered = p_expanded - p_hat[:, :, np.newaxis, :] # (H, W, N, 2)
    q_centered = q_expanded - q_hat[:, :, np.newaxis, :] # (H, W, N, 2)
    
    # 4. 计算仿射变换矩阵
    # MLS 仿射变形的公式涉及两个矩阵的乘积和逆
    q_outer = q_centered[:, :, :, :, np.newaxis]
    p_outer_T = p_centered[:, :, :, np.newaxis, :]
    weighted_outer_num = weights[:, :, :, np.newaxis, np.newaxis] * (q_outer @ p_outer_T)
    num_matrix = np.sum(weighted_outer_num, axis=-3) # (H, W, 2, 2)
    
    p_outer = p_centered[:, :, :, :, np.newaxis]
    p_outer_T_2 = p_centered[:, :, :, np.newaxis, :]
    weighted_outer_den = weights[:, :, :, np.newaxis, np.newaxis] * (p_outer @ p_outer_T_2)
    den_matrix = np.sum(weighted_outer_den, axis=-3) # (H, W, 2, 2)
    
    det = den_matrix[:, :, 0, 0] * den_matrix[:, :, 1, 1] - den_matrix[:, :, 0, 1] * den_matrix[:, :, 1, 0]
    det = np.maximum(np.abs(det), eps)
    
    inv_den = np.zeros_like(den_matrix)
    inv_den[:, :, 0, 0] = den_matrix[:, :, 1, 1] / det
    inv_den[:, :, 0, 1] = -den_matrix[:, :, 0, 1] / det
    inv_den[:, :, 1, 0] = -den_matrix[:, :, 1, 0] / det
    inv_den[:, :, 1, 1] = den_matrix[:, :, 0, 0] / det
    
    linear_part = num_matrix @ inv_den # (H, W, 2, 2)
    
    # 应用变换: f(v) = L * (v - p_hat) + q_hat
    v_minus_p_hat = v - p_hat # (H, W, 2)
    v_mp = v_minus_p_hat[:, :, :, np.newaxis]
    
    transformed_vec = linear_part @ v_mp # (H, W, 2, 1)
    transformed_vec = transformed_vec.squeeze(axis=-1) # (H, W, 2)
    
    source_coords = transformed_vec + q_hat # (H, W, 2)
    
    map_x = source_coords[:, :, 0].astype(np.float32)
    map_y = source_coords[:, :, 1].astype(np.float32)
    
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return warped_image

# 运行图像变形
def run_warping():
    global points_src, points_dst, image

    if image is None or len(points_src) == 0:
        return image
        
    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除所有选定的点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# 构建 Gradio 界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="上传图像", interactive=True, width=800)
            point_select = gr.Image(label="点击选择源点和目标点", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="变形结果", width=800)

    run_button = gr.Button("运行变形")
    clear_button = gr.Button("清除点")

    # 事件绑定
    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()