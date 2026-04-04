"""Small ResNet-18 for CIFAR-10.

Reduced channel widths [16, 32, 64, 128] instead of standard [64, 128, 256, 512].
~270K params. Uses BatchNorm and skip connections — expected to dampen init sensitivity.
"""

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SmallResNet18(nn.Module):
    def __init__(self, num_classes=10, widths=(16, 32, 64, 128)):
        super().__init__()
        self.in_planes = widths[0]

        self.conv1 = nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(widths[0])
        self.layer1 = self._make_layer(widths[0], 2, stride=1)
        self.layer2 = self._make_layer(widths[1], 2, stride=2)
        self.layer3 = self._make_layer(widths[2], 2, stride=2)
        self.layer4 = self._make_layer(widths[3], 2, stride=2)
        self.fc = nn.Linear(widths[3], num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)
