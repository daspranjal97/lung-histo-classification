
import torch
import torch.nn as nn
import torchvision.models as models

# Define Neighbor Feature Attention-based Pooling Layer
class NeighborFeatureAttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super(NeighborFeatureAttentionPooling, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, groups=in_channels)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        x = self.depthwise_conv(x * attn)
        return x

# Define the dual transfer learning model
class DualTransferLungClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(DualTransferLungClassifier, self).__init__()
        self.backbone1 = models.densenet169(pretrained=True).features
        self.backbone2 = models.inception_v3(pretrained=True, aux_logits=False).Conv2d_1a_3x3

        self.pool1 = NeighborFeatureAttentionPooling(1664)  # DenseNet final channels
        self.pool2 = NeighborFeatureAttentionPooling(32)    # InceptionV3 first layer output channels

        self.fc = nn.Sequential(
            nn.Linear(1664 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        feat1 = self.backbone1(x)
        feat1 = self.pool1(feat1)
        feat1 = torch.flatten(feat1, 1)

        feat2 = self.backbone2(x)
        feat2 = self.pool2(feat2)
        feat2 = torch.flatten(feat2, 1)

        combined = torch.cat((feat1, feat2), dim=1)
        out = self.fc(combined)
        return out
