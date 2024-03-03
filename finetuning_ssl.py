import torch
import torchvision
import torchvision.transforms as transforms
import time
import json
from uuid import uuid4
from typing import List
from collections import defaultdict
import os
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, RandomTranslate
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from cornet_s import CORnet_S
from cornet_rt import CORnet_RT
from collections import OrderedDict

from ffcvssl.loader import Loader as LoaderSSL
from ffcvssl.loader import OrderOption as OrderOptionSSL
from ffcvssl.colorjitter import RandomColorJitter
from ffcvssl.grayscale import RandomGrayscale
from losses import BYOL

import md_utils

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

CIFAR_MEAN = np.array([125.307, 122.961, 113.8575])
CIFAR_STD = np.array([51.5865, 50.847, 51.255])
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


Section('data', 'data related stuff').params(
    dataset_path=Param(str, '.dat file to use for training', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    results_folder=Param(str, 'where to save the data', required=True),
    old_log_path=Param(str, 'existing experiments', required=False, default=None)
)


Section('training', 'training related stuff').params(
    potential=Param(str, 'MD potential', required=True),
    n_d_dependency=Param(float, 'p so N = D^p; p=-1 means N = max N', required=True),
    xent_penalty=Param(float, 'L2 logit penalty on xent', required=True),
    pretrained=Param(bool, 'pretrained net or not', required=True)
)


@param('data.dataset_path')
@param('data.num_workers')
@param('data.in_memory')
def create_subset_val_loader_imagenet(dataset_path, num_workers, in_memory, *, batch_size=128,
                                      resolution=224, distributed=False, indices=None):
    this_device = 'cuda:0'
    val_path = Path(dataset_path)
    assert val_path.is_file()
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        RandomHorizontalFlip(),
        RandomGrayscale(0.2),
        RandomColorJitter(0.8, 0.4, 0.4, 0.4, 0.1),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    cropper_0 = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline_0 = [
        cropper_0,
        RandomHorizontalFlip(),
        RandomGrayscale(0.2),
        RandomColorJitter(0.8, 0.4, 0.4, 0.4, 0.1),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device),
                 non_blocking=True)
    ]

    # order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    order = OrderOptionSSL.RANDOM if distributed else OrderOptionSSL.QUASI_RANDOM
    # loader = Loader(dataset_path,
    #                 batch_size=batch_size,
    #                 num_workers=num_workers,
    #                 order=order,
    #                 drop_last=False,
    #                 os_cache=in_memory,
    #                 pipelines={
    #                     'image': image_pipeline,
    #                     'label': label_pipeline
    #                 },
    #                 distributed=distributed,
    #                 indices=indices)

    loader = LoaderSSL(dataset_path,
                       batch_size=batch_size,
                       num_workers=num_workers,
                       order=order,
                       os_cache=in_memory,
                       drop_last=False,
                       pipelines={
                           'image': image_pipeline,
                           'label': label_pipeline,
                           'image_0': image_pipeline_0
                       },
                       custom_field_mapper={'image_0': 'image'},  # FFCV-SSL specific
                       distributed=distributed,
                       indices=indices)

    return loader



# SimSiam code from
# https://github.com/facebookresearch/simsiam/blob/main/main_simsiam.py
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torchvision import models

class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z1 = self.encoder(x1) # NxC
        z2 = self.encoder(x2) # NxC

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

# facebook code ends

class ModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = SimSiam(models.resnet50)


def compute_accuracy(net, loader):
    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        with autocast():
            for idx, data in enumerate(loader):
                inputs = data[0].to('cuda:0')
                labels = data[1].to('cuda:0')
                # calculate outputs by running images through the network
                outputs = net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    return 100 * correct // total


def save_named_weights(net):
    saved_weights = dict()
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            saved_weights[name] = module.weight.detach().cpu()
    return saved_weights


def count_weights_only(net):
    n_weights = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            n_weights += module.weight.numel()
    return n_weights


def train_net(net, n_epochs, loader, optimizer, scheduler, criterion, scaler, dataset_length):
    net.train()

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(loader, 0): # trainloader
            # get the inputs; data is a list of [inputs, labels]
            inputs = data[0]#.to('cuda:0')
            labels = data[1]#.to('cuda:0')
            inputs_0 = data[2]

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # forward + backward + optimize
            with autocast():
                p1, p2, z1, z2 = net(inputs, inputs_0)
                loss = 0.5 * criterion(p1, z2) + 0.5 * criterion(p2, z1)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # print statistics
            running_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f'{epoch + 1} loss: {running_loss / dataset_length:.3f}')
        scheduler.step()

    return running_loss


@param('training.potential')
@param('training.n_d_dependency')
@param('training.xent_penalty')
@param('data.results_folder')
@param('training.pretrained')
def save_weights(init_weights, final_weights, arch, seed, potential, n_d_dependency, xent_penalty, results_folder,
                 pretrained):
    if seed == 0 or not pretrained:
        np.save(os.path.join(
            results_folder, f'{potential}_{arch}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}_init_{seed}'),
            init_weights)
    np.save(os.path.join(
        results_folder, f'{potential}_{arch}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}_final_{seed}'),
        final_weights)


@param('training.n_d_dependency')
def get_loader(n_params, arch, n_d_dependency):
    if n_d_dependency == -1:
        subset_ratio = 1
    else:
        subset_ratio = np.sqrt(n_params) / 50000
    if subset_ratio > 1:
        raise ValueError(f'{arch} D={n_params}, n_d_dependency{n_d_dependency}, ratio={subset_ratio} > 1!')

    if subset_ratio == 1.0:
        indices = None
    else:
        indices = np.random.permutation(50000)[:int(subset_ratio * 50000)].astype(int)
    return create_subset_val_loader_imagenet(resolution=224, batch_size=128, indices=indices)

@param('training.xent_penalty')
def get_criterion(xent_penalty):
    return BYOL()
    # criterion_xent = nn.CrossEntropyLoss()
    #
    # if xent_penalty == 0:
    #     criterion = criterion_xent
    # else:
    #     def criterion(x, y):
    #         return criterion_xent(x, y) + xent_penalty * (x ** 2).mean()
    # return criterion


@param('data.old_log_path')
def find_arch_to_rerun(old_log_path):
    arch_rerun = []
    if old_log_path is not None and os.path.isfile(old_log_path):
        print(f'LOADING OLD PATH AT {old_log_path}', flush=True)
        with open(old_log_path, 'r') as f:
            text = f.read()
        results = text.split('RUN SUMMARY\n\n\n')[1].split('\nJob finished')[0]
        for i, line in enumerate(results.split('\n')):
            if 'exit codes\t1' in line:
                arch_rerun.append(results.split('\n')[i - 1][:-1])
    return arch_rerun


@param('training.potential')
@param('training.pretrained')
def train(arch, lr, loader, dataset_length, n_epochs, potential, pretrained):
    if arch == 'resnet50':
        net_m = ModelWrapper()
        checkpoint_path = 'https://dl.fbaipublicfiles.com/simsiam/models/100ep/pretrain/checkpoint_0099.pth.tar'
        ckpt_data = torch.utils.model_zoo.load_url(checkpoint_path, map_location=None)
        net_m.load_state_dict(ckpt_data['state_dict'])
        net = net_m.module
    else:
        raise ValueError(arch)
    net.to('cuda:0')
    net.train()
    # for module in net.modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         if hasattr(module, 'weight'):
    #             module.weight.requires_grad_(False)
    #         if hasattr(module, 'bias'):
    #             module.bias.requires_grad_(False)
    #         module.eval()

    init_weights = save_named_weights(net)

    optimizer = md_utils.SGD(net.parameters(), lr=lr, momentum=0.9, potential=potential, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = get_criterion()
    scaler = GradScaler()

    final_loss = train_net(net, n_epochs, loader, optimizer, scheduler, criterion, scaler, dataset_length)
    accuracy = 0#compute_accuracy(net, loader)

    final_weights = save_named_weights(net)

    return final_loss, accuracy, init_weights, final_weights


def find_lr(arch, loader, dataset_length, n_epochs):
    lr_range = np.logspace(-5, -1, 5)
    losses = np.zeros_like(lr_range)
    accuracies = np.zeros_like(lr_range)

    for i, lr in enumerate(lr_range):
        losses[i], accuracies[i], _, _ = train(arch, lr, loader, dataset_length, n_epochs)
    idx_argmin = np.nanargmin(losses)
    exit_code = 1 if accuracies[idx_argmin] < 100 else 0
    return lr_range[idx_argmin], exit_code, losses[idx_argmin], accuracies[idx_argmin]


@param('training.potential')
@param('training.n_d_dependency')
@param('training.xent_penalty')
@param('training.pretrained')
@param('data.results_folder')
def main(potential, n_d_dependency, xent_penalty, pretrained, results_folder):
    n_seeds = 10
    arch_list = ['resnet50']

        # ['cornet-rt']#['cornet-s']#['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0']# ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'resnet18', 'resnet34']
    # ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'resnet18', 'resnet34', 'resnet50']
    exit_codes = dict()
    # n_bad_accuracy = defaultdict(int)
    # n_very_bad_accuracy = defaultdict(int)
    
    arch_to_rerun = [] # shouldn't work with ssl find_arch_to_rerun()
    print(F'ARCH TO RERUN {arch_to_rerun}', flush=True)
# ../../md_final_logs/finetuning_${POTENTIAL}_${ND}_${XENT}.out
# for POTENTIAL in '2-norm' 'negative_entropy' '3-norm'; do for ND in 0.5 0.33333; do for XENT in 0; do sbatch --gres=gpu:a100:1 --constraint="dgx&ampere" -c 4 --mem=15G -t 24:00:00 --partition=main --output="../../md_final_logs/finetuning_${POTENTIAL}_${ND}_${XENT}_restarted.out" --export=FOLDER="md_weights_neurips",POTENTIAL=$POTENTIAL,ND=$ND,XENT=$XENT,PR=$PR,RF=$RF,HOME=$HOME ./run_slurm_finetuning.sh; done; done; done
    for arch in arch_list:
        n_epochs = 50
        print(arch)
        if arch in arch_to_rerun or not os.path.isfile(os.path.join(
                results_folder,
                f'{potential}_{arch}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}_final_{9}.npy')):


            if arch == 'resnet50':
                net = ModelWrapper().module
            else:
                raise ValueError(arch)
            n_params = count_weights_only(net) #np.sum(np.fromiter((p.numel() for p in net.parameters() if p.requires_grad), dtype=int))
            del net
            loader = get_loader(n_params, arch)
            dataset_length = len(loader)


            optimal_lr, exit_codes[arch], final_loss, final_acc = find_lr(arch, loader, dataset_length, n_epochs)

            # accuracy doesn't really matter for SSL, just trying to minimize the loss
            # for i in range(4):
            #     if exit_codes[arch]:
            #         n_epochs += 30
            #         optimal_lr, exit_codes[arch], final_loss, final_acc = find_lr(arch, loader, dataset_length, n_epochs)
            # if exit_codes[arch]:
            print(f'Final loss {final_loss}, final acc {final_acc}, optimal lr {optimal_lr}',
                  flush=True)

            for seed in range(n_seeds):
                if arch in arch_to_rerun or not os.path.isfile(os.path.join(
                        results_folder,
                        f'{potential}_{arch}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}_final_{seed}.npy')):
                    final_loss, accuracy, init_weights, final_weights = train(arch, optimal_lr, loader,
                                                                              dataset_length, n_epochs)
                    # if accuracy < 100:
                    #     n_bad_accuracy[arch] += 1
                    #     if accuracy < 90:
                    #         n_very_bad_accuracy[arch] += 1
                    print(f'{arch} seed {seed} loss {final_loss}')
                    save_weights(init_weights, final_weights, arch, seed)

    print(f'RE-RUN ARCHES:\n{arch_to_rerun}')
    # print('RUN SUMMARY\n\n')
    # for arch in arch_list:
    #     print(f'{arch}:\n\texit codes\t{exit_codes[arch]}\n\t< 100 acc'
    #           f'\t{n_bad_accuracy[arch]}\n\t<90 acc\t{n_very_bad_accuracy[arch]}')


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast imagenet training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    main()
