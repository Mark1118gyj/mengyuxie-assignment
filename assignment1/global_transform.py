import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])

# Function to apply transformations based on user inputs
def apply_transform(image, scale, rotation, translation_x, translation_y, flip_horizontal):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros((pad_size*2+image.shape[0], pad_size*2+image.shape[1], 3), dtype=np.uint8) + np.array((255,255,255), dtype=np.uint8).reshape(1,1,3)
    image_new[pad_size:pad_size+image.shape[0], pad_size:pad_size+image.shape[1]] = image
    image = np.array(image_new)
    
    # 获取当前图像的高度和宽度
    h, w = image.shape[:2]
    # 计算图像中心
    center_x, center_y = w / 2, h / 2

    ### FILL: Apply Composition Transform 
    # Note: for scale and rotation, implement them around the center of the image （围绕图像中心进行放缩和旋转）

    # 1. 获取旋转和缩放的仿射矩阵 (2x3)
    # cv2.getRotationMatrix2D 中心点, 角度, 缩放比例
    # 注意：角度是逆时针为正，但通常UI中顺时针为正，这里假设输入rotation符合OpenCV习惯（逆时针）
    # 如果UI感觉方向反了，可以取负号 -rotation
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rotation, scale)

    # 2. 处理水平翻转
    # 水平翻转矩阵: [-1, 0, w; 0, 1, 0] 但这是相对于原点的。
    # 更好的方式是先构建翻转矩阵，然后与其他矩阵结合。
    # 简单方法：如果在最后应用翻转，需要小心坐标。
    # 让我们采用矩阵乘法的方式组合所有变换。
    
    # 将所有 2x3 矩阵转换为 3x3 以便乘法
    rot_mat_3x3 = to_3x3(rot_mat)
    
    # 构建平移矩阵 (Translation)
    # tx, ty 是用户输入的额外平移
    trans_mat = np.float32([[1, 0, translation_x],
                            [0, 1, translation_y]])
    trans_mat_3x3 = to_3x3(trans_mat)
    
    # 构建水平翻转矩阵 (Flip Horizontal)
    # 翻转通常是关于图像中心的 x=0 轴翻转，即 x' = -x + width
    # 矩阵: [[-1, 0, w], [0, 1, 0]]
    if flip_horizontal:
        flip_mat = np.float32([[-1, 0, w],
                               [0, 1, 0]])
    else:
        flip_mat = np.float32([[1, 0, 0],
                               [0, 1, 0]])
    flip_mat_3x3 = to_3x3(flip_mat)
    
    final_mat_3x3 = trans_mat_3x3 @ rot_mat_3x3 @ flip_mat_3x3
    
    # 转换回 2x3
    final_mat_2x3 = final_mat_3x3[:2, :]

    # 应用仿射变换
    transformed_image = cv2.warpAffine(image, final_mat_2x3, (w, h), borderValue=(255, 255, 255))

    return transformed_image

# Gradio Interface
def interactive_transform():
    with gr.Blocks() as demo:
        gr.Markdown("## Image Transformation Playground")
        
        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale")
                rotation = gr.Slider(minimum=-180, maximum=180, step=1, value=0, label="Rotation (degrees)")
                translation_x = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation X")
                translation_y = gr.Slider(minimum=-300, maximum=300, step=10, value=0, label="Translation Y")
                flip_horizontal = gr.Checkbox(label="Flip Horizontal")
            
            # Right: Output image
            image_output = gr.Image(label="Transformed Image")
        
        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input, scale, rotation, 
            translation_x, translation_y, 
            flip_horizontal
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)

    return demo

# Launch the Gradio interface
interactive_transform().launch()