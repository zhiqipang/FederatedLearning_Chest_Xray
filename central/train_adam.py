import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 添加项目根目录到路径，以便导入自定义模型
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn_model import PneumoniaCNN

#配置
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'chest_xray')
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './adam'  # 当前目录下的adam文件夹

print(f"Using device: {DEVICE}")
print(f"Optimizer: Adam, LR={LEARNING_RATE}, Epochs={EPOCHS}, Batch Size={BATCH_SIZE}")

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

#数据变换
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_datasets():
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)
    return train_dataset, test_dataset

def compute_class_weights(train_dataset):
    labels = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * 2
    return torch.tensor(weights, dtype=torch.float32)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc

#训练开始
start_time = time.time()

train_dataset, test_dataset = load_datasets()
class_weights = compute_class_weights(train_dataset).to(DEVICE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = PneumoniaCNN(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_accs = []
test_aucs = []

for epoch in range(1, EPOCHS+1):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
    acc, prec, rec, f1, auc = evaluate(model, test_loader)
    train_losses.append(train_loss)
    test_accs.append(acc)
    test_aucs.append(auc)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {train_loss:.4f} | Acc: {acc:.4f} | AUC: {auc:.4f}")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# 保存模型
model_path = os.path.join(SAVE_DIR, 'final_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# 保存训练曲线
plt.figure(figsize=(12, 5))
plt.subplot(1,2,1)
plt.plot(range(1, EPOCHS+1), test_accs, 'b-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Adam - Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(range(1, EPOCHS+1), test_aucs, 'r-', label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('Adam - AUC')
plt.legend()
plt.grid(True)

plt.tight_layout()
curve_path = os.path.join(SAVE_DIR, 'training_curves.png')
plt.savefig(curve_path)
plt.close()
print(f"Training curves saved to {curve_path}")

# 保存指标和总时长
final_metrics = {
    'accuracy': test_accs[-1],
    'precision': prec,
    'recall': rec,
    'f1': f1,
    'auc': test_aucs[-1],
    'total_time_seconds': total_time,
    'total_time_minutes': total_time/60
}
with open(os.path.join(SAVE_DIR, 'metrics.txt'), 'w') as f:
    f.write(f"Optimizer: Adam\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n\n")
    for k, v in final_metrics.items():
        if k not in ['total_time_seconds', 'total_time_minutes']:
            f.write(f"{k}: {v:.4f}\n")
    f.write(f"\ntotal_time_seconds: {total_time:.2f}\n")
    f.write(f"total_time_minutes: {total_time/60:.2f}\n")

print("\nFinal metrics:")
for k, v in final_metrics.items():
    if k == 'total_time_seconds':
        print(f"  total_time: {v:.2f} seconds")
    elif k == 'total_time_minutes':
        print(f"  total_time: {v:.2f} minutes")
    else:
        print(f"  {k}: {v:.4f}")