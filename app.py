import sys
import os
import gradio as gr
import torch
from torchvision import transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from models.cnn_model import PneumoniaCNN

MODEL_PATH = os.path.join(BASE_DIR, 'results', 'models', 'global_model_round_10.pth')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class_names = ['NORMAL', 'PNEUMONIA']

def predict(image):
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
    pred_class = class_names[predicted.item()]
    confidence_percent = confidence.item() * 100
    return f"诊断结论: {pred_class}", f"置信度: {confidence_percent:.2f}%"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传医疗影像"),
    outputs=[
        gr.Textbox(label="诊断结论"),
        gr.Textbox(label="置信度")
    ],
    title="基于联邦学习的医疗影像辅助诊断系统",
    description="上传一张胸部X光影像，模型将判断是否患有肺炎。",
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)