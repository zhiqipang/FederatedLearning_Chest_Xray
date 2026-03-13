import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):
    """
    一个简单的CNN模型，用于肺炎X光片二分类
    输入: (batch_size, 3, 224, 224)
    输出: (batch_size, 2) 对应 [正常, 肺炎] 的logits
    """
    def __init__(self, num_classes=2):
        super(PneumoniaCNN, self).__init__()
        # 卷积层1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # 输出: 112x112

        # 卷积层2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 输出: 56x56

        # 卷积层3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 输出: 28x28

        # 卷积层4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  # 输出: 14x14

        # 全局平均池化替代全连接，减少参数量
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 输出: 1x1

        # 分类器
        self.fc = nn.Linear(256, num_classes)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 卷积块1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        # 卷积块2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        # 卷积块3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        # 卷积块4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))

        # 全局平均池化
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # 展平

        # 分类
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# 简单测试
if __name__ == '__main__':
    model = PneumoniaCNN(num_classes=2)
    print(model)
    # 创建随机输入测试
    dummy_input = torch.randn(4, 3, 224, 224)  # batch_size=4
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")