import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import gradio as gr
from torchvision import transforms
from PIL import Image
from models.cnn_model import PneumoniaCNN

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练好的模型
model_path = 'results/models/global_model_final.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型文件未找到: {model_path}，请先运行 federated/server.py 训练模型。")

model = PneumoniaCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 定义与训练时相同的预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 类别名称
classes = ['NORMAL', 'PNEUMONIA']

def predict(image):
    """输入 PIL 图片，返回诊断结果和置信度"""
    # 预处理
    img = transform(image).unsqueeze(0).to(device)  # 添加 batch 维度

    # 推理
    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, pred = torch.max(probabilities, 0)
        pred_class = classes[pred.item()]
        confidence = conf.item()

    # 返回结果字符串和置信度
    if pred_class == 'PNEUMONIA':
        result = "肺炎阳性"
    else:
        result = "肺炎阴性"
    return result, f"{confidence:.2%}"

# 创建 Gradio 界面
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传X光片"),
    outputs=[
        gr.Textbox(label="诊断结果"),
        gr.Textbox(label="置信度")
    ],
    title="肺炎X光片辅助诊断系统",
    description="上传一张胸部X光片，模型将判断是否为肺炎。",
    examples=[],  # 可添加示例图片路径
)

if __name__ == '__main__':
    iface.launch()