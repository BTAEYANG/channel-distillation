import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import SGD

from cifar_config import Config
import losses
from models.channel_distillation import ChannelDistillResNet50152, ChannelDistillResNet_32x4_8x4, \
    ChannelDistillResNet_56_20, ChannelDistillResNet_110_32, ChannelDistillWrn_40_2_16_2, ChannelDistillWrn_40_2_40_1, \
    ChannelDistillVgg_13_8, ChannelDistillResnet32x4_shuffle_v1, ChannelDistillResnet32x4_shuffle_v2, \
    ChannelDistillVgg13_Mobile_v2
from utils.average_meter import AverageMeter
from utils.data_prefetcher import DataPrefetcher
from utils.logutil import get_logger
from utils.metric import accuracy
from utils.util import adjust_loss_alpha


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset
    parser.add_argument('--dataset', type=str, default='CIFAR100', choices=['CIFAR100', 'CIFAR10'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='resnet8',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4',
                                 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152',
                                 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'ResNet50',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2'])
    parser.add_argument('--path_t', type=str, default='./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth', help='teacher model snapshot')

    opt = parser.parse_args()

    return opt


def main():

    opt = parse_option()

    if not torch.cuda.is_available():
        raise Exception("need gpu to train network!")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    cudnn.benchmark = True
    cudnn.enabled = True

    logger = get_logger(__name__, Config.log)

    Config.gpus = torch.cuda.device_count()
    logger.info("use {} gpus".format(Config.gpus))
    config = {
        key: value
        for key, value in Config.__dict__.items() if not key.startswith("__")
    }
    # logger.info(f"args: {config}")

    start_time = time.time()

    # dataset and dataloader
    logger.info("start loading data")

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    train_dataset = CIFAR100(
        Config.train_dataset_path,
        train=True,
        transform=train_transform,
        download=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    val_dataset = CIFAR100(
        Config.val_dataset_path,
        train=False,
        transform=val_transform,
        download=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    logger.info("finish loading data")

    # network
    # net = ChannelDistillResNet50152(Config.num_classes, Config.dataset_type)
    if opt.model_s == 'resnet8x4':
        net = ChannelDistillResNet_32x4_8x4(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'resnet20':
        net = ChannelDistillResNet_56_20(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'resnet32':
        net = ChannelDistillResNet_110_32(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'wrn_16_2':
        net = ChannelDistillWrn_40_2_16_2(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'wrn_40_1':
        net = ChannelDistillWrn_40_2_40_1(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'vgg8':
        net = ChannelDistillVgg_13_8(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'ShuffleV1':
        net = ChannelDistillResnet32x4_shuffle_v1(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'ShuffleV2':
        net = ChannelDistillResnet32x4_shuffle_v2(num_classes=Config.num_classes, pth_path=opt.path_t)
    elif opt.model_s == 'MobileNetV2':
        net = ChannelDistillVgg13_Mobile_v2(num_classes=Config.num_classes, pth_path=opt.path_t)
    net = nn.DataParallel(net).cuda()

    # loss and optimizer
    criterion = []
    for loss_item in Config.loss_list:
        loss_name = loss_item["loss_name"]
        loss_type = loss_item["loss_type"]
        if "kd" in loss_type:
            criterion.append(losses.__dict__[loss_name](loss_item["T"]).cuda())
        else:
            criterion.append(losses.__dict__[loss_name]().cuda())

    optimizer = SGD(net.parameters(), lr=Config.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # only evaluate
    if Config.evaluate:
        # load best model
        if not os.path.isfile(Config.evaluate):
            raise Exception(f"{Config.evaluate} is not a file, please check it again")
        logger.info("start evaluating")
        logger.info(f"start resuming model from {Config.evaluate}")
        checkpoint = torch.load(Config.evaluate, map_location=torch.device("cpu"))
        net.load_state_dict(checkpoint["model_state_dict"])
        prec1, prec5 = validate(val_loader, net)
        logger.info(f"epoch {checkpoint['epoch']:0>3d}, top1 acc: {prec1:.2f}%, top5 acc: {prec5:.2f}%")
        return

    start_epoch = 1
    # resume training
    # if os.path.exists(Config.resume):
    #     logger.info(f"start resuming model from {Config.resume}")
    #     checkpoint = torch.load(Config.resume, map_location=torch.device("cpu"))
    #     start_epoch += checkpoint["epoch"]
    #     net.load_state_dict(checkpoint["model_state_dict"])
    #     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #     scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    #     logger.info(
    #         f"finish resuming model from {Config.resume}, epoch {checkpoint['epoch']}, "
    #         f"loss: {checkpoint['loss']:3f}, lr: {checkpoint['lr']:.6f}, "
    #         f"top1_acc: {checkpoint['acc']}%, loss {checkpoint['loss']}%"
    #     )

    if not os.path.exists(Config.checkpoints):
        os.makedirs(Config.checkpoints)

    logger.info("start training")
    best_acc = 0.
    for epoch in range(start_epoch, Config.epochs + 1):
        prec1, prec5, loss = train(train_loader, net, criterion, optimizer, scheduler,
                                   epoch, logger)
        logger.info(f"----train----: epoch {epoch:0>3d}, top1 acc: {prec1:.2f}%, top5 acc: {prec5:.2f}% \t\n")

        prec1, prec5 = validate(val_loader, net)
        logger.info(f"====val====: epoch {epoch:0>3d}, top1 acc: {prec1:.2f}%, top5 acc: {prec5:.2f}% \t\n")

        # remember best prec@1 and save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "acc": prec1,
                "loss": loss,
                "lr": scheduler.get_last_lr()[0],
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            os.path.join(Config.checkpoints, "latest.pth")
        )
        if prec1 > best_acc:
            shutil.copyfile(os.path.join(Config.checkpoints, "latest.pth"),
                            os.path.join(Config.checkpoints, "best.pth"))
            best_acc = prec1

    training_time = (time.time() - start_time) / 3600
    logger.info(f"finish training, best acc: {best_acc:.2f}%, total training time: {training_time:.2f} hours")


def train(train_loader, net, criterion, optimizer, scheduler, epoch, logger):
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_total = AverageMeter()

    loss_ams = [AverageMeter()] * len(criterion)
    loss_alphas = []
    for loss_item in Config.loss_list:
        loss_rate = loss_item["loss_rate"]
        factor = loss_item["factor"]
        loss_type = loss_item["loss_type"]
        loss_rate_decay = loss_item["loss_rate_decay"]
        loss_alphas.append(
            adjust_loss_alpha(loss_rate, epoch, factor, loss_type, loss_rate_decay, Config.dataset_type)
        )

    # switch to train mode
    net.train()

    iters = len(train_loader.dataset) // Config.batch_size
    prefetcher = DataPrefetcher(train_loader)
    inputs, labels = prefetcher.next()
    iter = 1
    while inputs is not None:
        inputs, labels = inputs.float().cuda(), labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        stu_outputs, tea_outputs = net(inputs)
        loss = 0
        loss_detail = []
        for i, loss_item in enumerate(Config.loss_list):
            loss_type = loss_item["loss_type"]
            if loss_type == "ce_family":
                tmp_loss = loss_alphas[i] * criterion[i](stu_outputs[-1], labels)
            elif loss_type == "kd_family":
                tmp_loss = loss_alphas[i] * criterion[i](stu_outputs[-1], tea_outputs[-1])
            elif loss_type == "kdv2_family":
                tmp_loss = loss_alphas[i] * criterion[i](stu_outputs[-1], tea_outputs[-1], labels)
            elif loss_type == "fd_family":
                tmp_loss = loss_alphas[i] * criterion[i](stu_outputs[:-2], tea_outputs[:-2])

            loss_detail.append(tmp_loss.item())
            loss_ams[i].update(tmp_loss.item(), inputs.size(0))
            loss += tmp_loss

        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(stu_outputs[-1], labels, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
        loss_total.update(loss.item(), inputs.size(0))

        inputs, labels = prefetcher.next()

        if iter % 20 == 0:
            loss_log = f"epoch: {epoch:0>3d}, iter: [{iter:0>4d}/{iters:0>4d}], lr: {scheduler.get_last_lr()[0]:.5f} "
            loss_log += f"top1 acc: {prec1.item():.2f}%, top5 acc: {prec5.item():.2f}% "
            loss_log += f"loss_total: {loss.item():3f} "
            # for i, loss_item in enumerate(Config.loss_list):
            #     loss_name = loss_item["loss_name"]
            #     loss_log += f"{loss_name}: {loss_detail[i]:3f}, alpha: {loss_alphas[i]:3f}, "
            logger.info(loss_log)

        iter += 1

    scheduler.step()

    return top1.avg, top5.avg, loss_total.avg


def validate(val_loader, net):
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    net.eval()

    prefetcher = DataPrefetcher(val_loader)
    inputs, labels = prefetcher.next()
    with torch.no_grad():
        while inputs is not None:
            inputs = inputs.float().cuda()
            labels = labels.cuda()

            stu_outputs, _ = net(inputs)

            pred1, pred5 = accuracy(stu_outputs[-1], labels, topk=(1, 5))
            top1.update(pred1.item(), inputs.size(0))
            top5.update(pred5.item(), inputs.size(0))
            inputs, labels = prefetcher.next()

    return top1.avg, top5.avg


if __name__ == "__main__":
    main()
