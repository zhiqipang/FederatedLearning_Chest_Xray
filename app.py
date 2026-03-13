import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
from models.cnn_model import PneumoniaCNN

# 配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'results/models/global_model_final.pth'
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# 加载模型
model = PneumoniaCNN().to(DEVICE)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"模型已加载: {MODEL_PATH}")
else:
    print(f"警告: 模型文件 {MODEL_PATH} 不存在，将使用随机初始化模型")

# 定义预处理（与训练时保持一致）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def predict(image):
    """输入 PIL Image，返回诊断结果和置信度"""
    # 预处理
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)  # 添加 batch 维度

    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[pred_class].item()

    result = CLASS_NAMES[pred_class]
    return result, f"{confidence:.2%}"


# 创建 Gradio 界面
with gr.Blocks(title="肺炎X光片辅助诊断系统") as demo:
    gr.Markdown("# 基于联邦学习的肺炎X光片辅助诊断系统")
    gr.Markdown("上传一张胸部X光片（PNG/JPG），模型将判断是否为肺炎。")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传X光片")
            diagnose_btn = gr.Button("开始诊断")
        with gr.Column():
            result_output = gr.Textbox(label="诊断结果")
            confidence_output = gr.Textbox(label="置信度")

    diagnose_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[result_output, confidence_output]
    )

    gr.Markdown("---")
    gr.Markdown("**说明**：本系统基于联邦学习训练，保护数据隐私。结果仅供参考，不构成医疗建议。")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)