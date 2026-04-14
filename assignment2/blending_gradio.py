import gradio as gr
from PIL import ImageDraw, Image
import numpy as np
import torch
import torch.nn.functional as F

# 初始化多边形状态
def initialize_polygon():
    return {'points': [], 'closed': False}

# 点击添加点
def add_point(img_original, polygon_state, evt: gr.SelectData):
    if polygon_state['closed']:
        return img_original, polygon_state

    x, y = evt.index
    polygon_state['points'].append((x, y))

    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)

    # 绘制连线
    if len(polygon_state['points']) > 1:
        draw.line(polygon_state['points'], fill='red', width=2)

    # 绘制顶点
    for point in polygon_state['points']:
        draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill='blue')

    return img_with_poly, polygon_state

# 闭合多边形
def close_polygon(img_original, polygon_state):
    if not polygon_state['closed'] and len(polygon_state['points']) > 2:
        polygon_state['closed'] = True
        img_with_poly = img_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        draw.polygon(polygon_state['points'], outline='red')
        return img_with_poly, polygon_state
    else:
        return img_original, polygon_state

# 更新背景上的多边形预览
def update_background(background_image_original, polygon_state, dx, dy):
    if background_image_original is None:
        return None

    if polygon_state['closed']:
        img_with_poly = background_image_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        shifted_points = [(x + dx, y + dy) for x, y in polygon_state['points']]
        draw.polygon(shifted_points, outline='red')
        return img_with_poly
    else:
        return background_image_original

# 从点集创建二值掩膜
def create_mask_from_points(points, img_h, img_w):
    """
    创建二值掩膜：内部255，外部0
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    
    # FILL: 使用 PIL 绘制填充多边形
    if len(points) > 2:
        pil_mask = Image.new('L', (img_w, img_h), 0)
        draw = ImageDraw.Draw(pil_mask)
        # 转换点格式为 tuple list
        poly_points = [tuple(p) for p in points]
        draw.polygon(poly_points, fill=255)
        mask = np.array(pil_mask)

    return mask

# 计算拉普拉斯损失
def cal_laplacian_loss(foreground_img, blended_img, mask):
    """
    计算掩膜区域内的拉普拉斯算子差异损失
    注意：foreground_img 和 blended_img 必须具有相同的形状 (B, C, H, W)
    mask 形状为 (B, 1, H, W) 或 (B, C, H, W)
    """
    # 定义拉普拉斯核
    laplacian_kernel = torch.tensor([[0, 1, 0], 
                                     [1, -4, 1], 
                                     [0, 1, 0]], 
                                    dtype=torch.float32, 
                                    device=foreground_img.device).unsqueeze(0).unsqueeze(0)
    
    B, C, H, W = foreground_img.shape
    
    # 重塑以应用单通道卷积到每个通道 (B*C, 1, H, W)
    fg_reshaped = foreground_img.view(B * C, 1, H, W)
    bl_reshaped = blended_img.view(B * C, 1, H, W)
    
    # 计算拉普拉斯响应 (padding=1 保持尺寸)
    fg_lap = F.conv2d(fg_reshaped, laplacian_kernel, padding=1)
    bl_lap = F.conv2d(bl_reshaped, laplacian_kernel, padding=1)
    
    # 恢复形状
    fg_lap = fg_lap.view(B, C, H, W)
    bl_lap = bl_lap.view(B, C, H, W)
    
    # 计算差异
    diff = fg_lap - bl_lap
    
    # 确保 mask 形状匹配 diff (B, C, H, W)
    if mask.shape[1] == 1:
        mask_expanded = mask.expand_as(diff)
    else:
        mask_expanded = mask
        
    # Masked MSE Loss
    loss = torch.sum((diff * mask_expanded) ** 2)
    
    # 归一化
    num_pixels = torch.sum(mask_expanded)
    if num_pixels > 0:
        loss = loss / num_pixels

    return loss

# 执行泊松融合
def blending(foreground_image_original, background_image_original, dx, dy, polygon_state):
    if not polygon_state['closed'] or background_image_original is None or foreground_image_original is None:
        return background_image_original

    # 转换为 numpy
    foreground_np = np.array(foreground_image_original)
    background_np = np.array(background_image_original)

    # 获取偏移后的多边形点
    foreground_polygon_points = np.array(polygon_state['points']).astype(np.int64)
    background_polygon_points = foreground_polygon_points + np.array([int(dx), int(dy)]).reshape(1, 2)

    # 创建掩膜
    foreground_mask = create_mask_from_points(foreground_polygon_points, foreground_np.shape[0], foreground_np.shape[1])
    background_mask = create_mask_from_points(background_polygon_points, background_np.shape[0], background_np.shape[1])

    # 转换为 Tensor
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    fg_img_tensor = torch.from_numpy(foreground_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    bg_img_tensor = torch.from_numpy(background_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    
    # 掩膜 Tensor (0 or 1)
    fg_mask_tensor = torch.from_numpy(foreground_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.
    bg_mask_tensor = torch.from_numpy(background_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.

    # --- 关键修改：将前景图像和掩膜嵌入到与背景相同大小的张量中 ---
    B, C, H_bg, W_bg = bg_img_tensor.shape
    _, _, H_fg, W_fg = fg_img_tensor.shape
    
    # 创建与背景同尺寸的全零张量用于放置前景
    fg_img_padded = torch.zeros_like(bg_img_tensor)
    fg_mask_padded = torch.zeros_like(bg_mask_tensor)
    
    # 计算放置位置的边界
    # 原始点在前景图中的坐标是多边形点
    # 背景中的坐标是 多边形点 + dx, dy
    # 我们需要确定前景图在背景图中的映射关系
    # 简单假设：前景图的左上角对应背景图的 (dx, dy) ? 
    # 不，多边形点是相对于前景图左上角的。
    # 背景中的多边形点 = 前景多边形点 + offset(dx, dy)
    # 这意味着前景图像素 (u, v) 对应背景图像素 (u + dx, v + dy)
    
    # 我们需要找到重叠区域
    # 前景范围: [0, W_fg), [0, H_fg)
    # 对应背景范围: [dx, dx + W_fg), [dy, dy + H_fg)
    
    # 计算交集
    start_x_bg = max(0, dx)
    end_x_bg = min(W_bg, dx + W_fg)
    start_y_bg = max(0, dy)
    end_y_bg = min(H_bg, dy + H_fg)
    
    start_x_fg = start_x_bg - dx
    end_x_fg = end_x_bg - dx
    start_y_fg = start_y_bg - dy
    end_y_fg = end_y_bg - dy
    
    # 确保索引有效
    if start_x_fg < end_x_fg and start_y_fg < end_y_fg:
        fg_img_padded[:, :, start_y_bg:end_y_bg, start_x_bg:end_x_bg] = fg_img_tensor[:, :, start_y_fg:end_y_fg, start_x_fg:end_x_fg]
        fg_mask_padded[:, :, start_y_bg:end_y_bg, start_x_bg:end_x_bg] = fg_mask_tensor[:, :, start_y_fg:end_y_fg, start_x_fg:end_x_fg]
    
    # 现在 fg_img_padded 和 bg_img_tensor 尺寸相同
    # 初始化混合图像
    blended_img = bg_img_tensor.clone()
    
    # 仅在 mask 区域内初始化前景内容 (可选，有助于收敛)
    mask_bool = bg_mask_tensor.bool().expand(-1, 3, -1, -1)
    try:
        # 使用嵌入后的前景图进行初始化
        blended_img[mask_bool] = fg_img_padded[mask_bool]
    except:
        pass
        
    blended_img.requires_grad = True

    # 优化器
    optimizer = torch.optim.Adam([blended_img], lr=1e-2)

    # 优化循环
    iter_num = 5000
    for step in range(iter_num):
        # 构造用于计算损失的图像：外部是背景(detach)，内部是可优化的 blended_img
        # 这样梯度只通过 mask 内部传播
        blended_img_for_loss = blended_img.detach() * (1. - bg_mask_tensor) + blended_img * bg_mask_tensor

        # 使用嵌入后的前景图和掩膜计算损失
        # 注意：这里我们比较的是 blended_img_for_loss 和 fg_img_padded 在 bg_mask_tensor 区域内的拉普拉斯响应
        loss = cal_laplacian_loss(fg_img_padded, blended_img_for_loss, bg_mask_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f'步骤: {step}, 损失: {loss.item()}')

        if step == int(iter_num*2/3): 
            optimizer.param_groups[0]['lr'] *= 0.1

    # 输出结果
    result = torch.clamp(blended_img.detach(), 0, 1).cpu().permute(0, 2, 3, 1).squeeze().numpy() * 255
    result = result.astype(np.uint8)
    return result

# 辅助函数：闭合并重置
def close_polygon_and_reset_dx(img_original, polygon_state, dx, dy, background_image_original):
    img_with_poly, updated_polygon_state = close_polygon(img_original, polygon_state)
    new_dx = gr.update(value=0)
    updated_background = update_background(background_image_original, updated_polygon_state, 0, dy)
    return img_with_poly, updated_polygon_state, updated_background, new_dx

# Gradio 界面
with gr.Blocks(title="Poisson Image Blending", css="""
    body { background-color: #1e1e1e; color: #ffffff; }
    .gr-button { background-color: #6200ee; color: #ffffff; border: none; }
    .gr-slider input[type=range] { accent-color: #03dac6; }
""") as demo:
    polygon_state = gr.State(initialize_polygon())
    background_image_original = gr.State(value=None)

    gr.Markdown("<h1 style='text-align: center;'>泊松图像融合 (Poisson Blending)</h1>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 前景图 (绘制多边形)")
            foreground_image_original = gr.Image(type="pil", interactive=True, height=300)
            foreground_image_with_polygon = gr.Image(type="pil", interactive=True, height=300)
            close_polygon_button = gr.Button("闭合多边形")
        with gr.Column():
            gr.Markdown("### 背景图")
            background_image = gr.Image(type="pil", interactive=True, height=300)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 背景预览 (带偏移)")
            background_image_with_polygon = gr.Image(type="pil", height=500)
        with gr.Column():
            gr.Markdown("### 融合结果")
            output_image = gr.Image(type="pil", height=500)

    with gr.Row():
        dx = gr.Slider(label="水平偏移", minimum=-500, maximum=500, step=1, value=0)
        dy = gr.Slider(label="垂直偏移", minimum=-500, maximum=500, step=1, value=0)
        blend_button = gr.Button("开始融合")

    # 事件绑定
    foreground_image_original.change(fn=lambda img: img, inputs=foreground_image_original, outputs=foreground_image_with_polygon)
    foreground_image_with_polygon.select(add_point, inputs=[foreground_image_original, polygon_state], outputs=[foreground_image_with_polygon, polygon_state])
    close_polygon_button.click(close_polygon_and_reset_dx, inputs=[foreground_image_original, polygon_state, dx, dy, background_image_original], outputs=[foreground_image_with_polygon, polygon_state, background_image_with_polygon, dx])
    background_image.change(fn=lambda img: img, inputs=background_image, outputs=background_image_original)
    dx.change(update_background, inputs=[background_image_original, polygon_state, dx, dy], outputs=background_image_with_polygon)
    dy.change(update_background, inputs=[background_image_original, polygon_state, dx, dy], outputs=background_image_with_polygon)
    blend_button.click(blending, inputs=[foreground_image_original, background_image_original, dx, dy, polygon_state], outputs=output_image)

demo.launch()