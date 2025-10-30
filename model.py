# import torch.nn as nn
# import torchvision.models as models

# class ResNet50(nn.Module):
#     def __init__(self, num_classes):
#         super(ResNet50, self).__init__()
#         self.model = models.resnet50(weights=None)
#         self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.model(x)

import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        # TorchVision 0.19+ uses Weights enums
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.resnet50(weights=weights)
        in_f = self.model.fc.in_features
        self.model.fc = nn.Linear(in_f, num_classes)

    def forward(self, x):
        return self.model(x)
