import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b4, EfficientNet_B4_Weights, efficientnet_b3, EfficientNet_B3_Weights, efficientnet_v2_m, EfficientNet_V2_M_Weights, resnext50_32x4d, ResNeXt50_32X4D_Weights, efficientnet_v2_l, EfficientNet_V2_L_Weights, EfficientNet_V2_S_Weights, efficientnet_v2_s, Swin_V2_T_Weights, swin_v2_t

from base import BaseModel



class ResNet50Model(BaseModel):
    def __init__(self, output_dim=10):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, output_dim, bias=True)

    def forward(self, x):
        x = self.features(x)
        x = x.squeeze()
        x = self.fc(x)
        return x


class ResNet50TemperatureModel(BaseModel):
    def __init__(self, output_dim=10):
        super().__init__()
        self.net = ResNet50Model(output_dim + 1)


    def forward(self, x):
        x = self.net(x)

        logits, temperature = x.split([x.size(1) - 1, 1], dim=1)
        temperature = temperature.squeeze()

        return logits, temperature




class EfficientNetB3Model(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-2])
        lastconv_output_channels = 1536

        self.avgpool = nn.AdaptiveAvgPool2d(final_spatial_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels * final_spatial_size * final_spatial_size, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class EfficientNetB3TemperatureModel(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        self.efficient_net = EfficientNetB3Model(output_dim + 1, final_spatial_size, dropout)


    def forward(self, x):
        x = self.efficient_net(x)

        logits, temperature = x.split([x.size(1) - 1, 1], dim=1)
        temperature = temperature.squeeze()

        return logits, temperature


class EfficientNetV2MModel(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-2])
        lastconv_output_channels = 1280

        self.avgpool = nn.AdaptiveAvgPool2d(final_spatial_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels * final_spatial_size * final_spatial_size, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class EfficientNetV2MTemperatureModel(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        self.efficient_net = EfficientNetV2MModel(output_dim + 1, final_spatial_size, dropout)

    def forward(self, x):
        x = self.efficient_net(x)

        logits, temperature = x.split([x.size(1) - 1, 1], dim=1)
        temperature = temperature.squeeze()

        return logits, temperature




class EfficientNetV2SModel(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-2])
        lastconv_output_channels = 1280

        self.avgpool = nn.AdaptiveAvgPool2d(final_spatial_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels * final_spatial_size * final_spatial_size, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




class EfficientNetV2STemperatureModel(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        self.efficient_net = EfficientNetV2SModel(output_dim + 1, final_spatial_size, dropout)

    def forward(self, x):
        x = self.efficient_net(x)

        logits, temperature = x.split([x.size(1) - 1, 1], dim=1)
        temperature = temperature.squeeze()

        return logits, temperature





class EfficientNetV2LModel(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-2])
        lastconv_output_channels = 1280

        self.avgpool = nn.AdaptiveAvgPool2d(final_spatial_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels * final_spatial_size * final_spatial_size, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNeXt50Model(BaseModel):
    def __init__(self, output_dim=10, final_spatial_size=1, dropout=0.1):
        super().__init__()
        model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
        self.features = nn.Sequential(*list(model.children())[:-2])
        lastconv_output_channels = 2048

        self.avgpool = nn.AdaptiveAvgPool2d(final_spatial_size)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(lastconv_output_channels * final_spatial_size * final_spatial_size, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x





class ResNeXt50TemperatureModel(BaseModel):
    def __init__(self, output_dim=10):
        super().__init__()
        self.net = ResNeXt50Model(output_dim + 1)

    def forward(self, x):
        x = self.net(x)

        logits, temperature = x.split([x.size(1) - 1, 1], dim=1)
        temperature = temperature.squeeze()

        return logits, temperature





class SwinV2TModel(BaseModel):
    def __init__(self, output_dim=10):
        super().__init__()
        model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(model.children())[:-1])
        lastconv_output_channels = 768
        self.classifier = nn.Linear(lastconv_output_channels, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



class SwinV2TTemperatureModel(BaseModel):
    def __init__(self, output_dim=10):
        super().__init__()
        self.swin_net = SwinV2TModel(output_dim + 1)

    def forward(self, x):
        x = self.swin_net(x)

        logits, temperature = x.split([x.size(1) - 1, 1], dim=1)
        temperature = temperature.squeeze()

        return logits, temperature



