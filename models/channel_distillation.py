import torch
import torch.nn as nn

from . import model_dict, wrn_16_2, wrn_40_1
from .ShuffleNetv1 import ShuffleV1
from .resnet import resnet18, resnet34, resnet50, resnet152
from .resnext import resnet20, resnet32x4, resnet8x4, resnet56, resnet32
from .vgg import vgg8


def conv1x1_bn(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class ChannelDistillResNet1834(nn.Module):

    def __init__(self, num_classes=1000, dataset_type="imagenet"):
        super().__init__()
        self.student = resnet18(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)
        self.teacher = resnet34(True, num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)

        self.s_t_pair = [(64, 64), (128, 128), (256, 256), (512, 512)]
        self.connector = nn.ModuleList([conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x)
        ts = self.teacher(x)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillResNet50152(nn.Module):
    def __init__(self, num_classes=100, dataset_type="imagenet"):
        super().__init__()
        self.student = resnet50(num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)
        self.teacher = resnet152(True, num_classes=num_classes, inter_layer=True, dataset_type=dataset_type)

        self.s_t_pair = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x)
        ts = self.teacher(x)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path)['model'])
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model'])
    print('==> done')
    return model


class ChannelDistillResNet_56_20(nn.Module):
    def __init__(self, num_classes=100, pth_path=''):
        super().__init__()
        self.student = resnet20(num_classes=num_classes)
        self.teacher = load_teacher(model_path=pth_path, n_cls=num_classes)

        self.s_t_pair = [(16, 16), (16, 16), (32, 32), (64, 64)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x, is_feat=True, preact=False)
        ts = self.teacher(x, is_feat=True, preact=False)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillResNet_110_32(nn.Module):
    def __init__(self, num_classes=100, pth_path=''):
        super().__init__()
        self.student = resnet32(num_classes=num_classes)
        self.teacher = load_teacher(model_path=pth_path, n_cls=num_classes)

        self.s_t_pair = [(16, 16), (16, 16), (32, 32), (64, 64)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x, is_feat=True, preact=False)
        ts = self.teacher(x, is_feat=True, preact=False)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillResNet_32x4_8x4(nn.Module):
    def __init__(self, num_classes=100, pth_path=''):
        super().__init__()
        self.student = resnet8x4(num_classes=num_classes)
        self.teacher = load_teacher(model_path=pth_path, n_cls=num_classes)

        self.s_t_pair = [(32, 32), (64, 64), (128, 128), (256, 256)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x, is_feat=True, preact=False)
        ts = self.teacher(x, is_feat=True, preact=False)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillWrn_40_2_16_2(nn.Module):
    def __init__(self, num_classes=100, pth_path=''):
        super().__init__()
        self.student = wrn_16_2(num_classes=num_classes)
        self.teacher = load_teacher(model_path=pth_path, n_cls=num_classes)

        self.s_t_pair = [(16, 16), (32, 32), (64, 64), (128, 128)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x, is_feat=True, preact=False)
        ts = self.teacher(x, is_feat=True, preact=False)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillWrn_40_2_40_1(nn.Module):
    def __init__(self, num_classes=100, pth_path=''):
        super().__init__()
        self.student = wrn_40_1(num_classes=num_classes)
        self.teacher = load_teacher(model_path=pth_path, n_cls=num_classes)

        self.s_t_pair = [(16, 16), (16, 32), (32, 64), (64, 128)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x, is_feat=True, preact=False)
        ts = self.teacher(x, is_feat=True, preact=False)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillVgg_13_8(nn.Module):
    def __init__(self, num_classes=100, pth_path=''):
        super().__init__()
        self.student = vgg8(num_classes=num_classes)
        self.teacher = load_teacher(model_path=pth_path, n_cls=num_classes)

        self.s_t_pair = [(64, 64), (128, 128), (256, 256), (512, 512), (512, 512)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x, is_feat=True, preact=False)
        ts = self.teacher(x, is_feat=True, preact=False)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts


class ChannelDistillResnet32x4_shuffle_v1(nn.Module):
    def __init__(self, num_classes=100, pth_path=''):
        super().__init__()
        self.student = ShuffleV1(num_classes=num_classes)
        self.teacher = load_teacher(model_path=pth_path, n_cls=num_classes)

        self.s_t_pair = [(24, 32), (240, 64), (480, 128), (960, 256)]
        self.connector = nn.ModuleList(
            [conv1x1_bn(s, t) for s, t in self.s_t_pair])
        # freeze teacher
        for m in self.teacher.parameters():
            m.requires_grad = False

    def forward(self, x):
        ss = self.student(x, is_feat=True, preact=False)
        ts = self.teacher(x, is_feat=True, preact=False)
        for i in range(len(self.s_t_pair)):
            ss[i] = self.connector[i](ss[i])

        return ss, ts
