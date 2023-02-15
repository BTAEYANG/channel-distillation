from models.resnext import resnet20, resnet56, resnet8x4, resnet32x4, resnet32, resnet110
from models.vgg import vgg8_bn, vgg13_bn
from models.wrn import wrn_40_1, wrn_16_2, wrn_40_2

model_dict = {
    # 'resnet8': resnet8,
    # 'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    # 'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    # 'ResNet18': ResNet18,
    # 'ResNet34': ResNet34,
    # 'ResNet50': ResNet50,
    # 'ResNet101': ResNet101,
    # 'ResNet152': ResNet152,
    # 'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    # 'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    # 'vgg16': vgg16_bn,
    # 'vgg19': vgg19_bn,
    # 'MobileNetV2': mobile_half,
    # 'ShuffleV1': ShuffleV1,
    # 'ShuffleV2': ShuffleV2,
}