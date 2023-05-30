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

import md_utils

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


Section('data', 'data related stuff').params(
    dataset_path=Param(str, '.dat file to use for training', required=True),
    num_workers=Param(int, 'The number of workers', required=True),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),
    results_folder=Param(str, 'where to save the data', required=True)
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
def create_subset_val_loader_imagenet(dataset_path, num_workers, in_memory, *, batch_size=256,
                                      resolution=224, distributed=False, indices=None):
    this_device = 'cuda:0'
    val_path = Path(dataset_path)
    assert val_path.is_file()
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
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

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loader = Loader(dataset_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    drop_last=False,
                    os_cache=in_memory,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed,
                    indices=indices)
    return loader


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

            # zero the parameter gradients
            optimizer.zero_grad(set_to_none=True)

            # forward + backward + optimize
            with autocast():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
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
        subset_ratio = n_params ** n_d_dependency / 50000
    if subset_ratio > 1:
        raise ValueError(f'{arch} D={n_params}, n_d_dependency{n_d_dependency}, ratio={subset_ratio} > 1!')

    if subset_ratio == 1.0:
        indices = None
    else:
        indices = np.random.permutation(50000)[:int(subset_ratio * 50000)].astype(int)
    return create_subset_val_loader_imagenet(resolution=224, batch_size=256, indices=indices)


@param('training.xent_penalty')
def get_criterion(xent_penalty):
    criterion_xent = nn.CrossEntropyLoss()

    if xent_penalty == 0:
        criterion = criterion_xent
    else:
        def criterion(x, y):
            return criterion_xent(x, y) + xent_penalty * (x ** 2).mean()
    return criterion


@param('training.potential')
@param('training.pretrained')
def train(arch, lr, loader, dataset_length, n_epochs, potential, pretrained):
    net = getattr(models, arch)(pretrained=pretrained)
    net.to('cuda:0')
    net.train()

    init_weights = save_named_weights(net)

    optimizer = md_utils.SGD(net.parameters(), lr=lr, momentum=0.9, potential=potential, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = get_criterion()
    scaler = GradScaler()

    final_loss = train_net(net, n_epochs, loader, optimizer, scheduler, criterion, scaler, dataset_length)
    accuracy = compute_accuracy(net, loader)

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
    arch_list = ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'efficientnet_b0', 'efficientnet_b1',
                 'efficientnet_b2', 'efficientnet_b3', 'resnet18', 'resnet34']
    exit_codes = dict()
    n_bad_accuracy =  defaultdict(int)
    n_very_bad_accuracy =  defaultdict(int)

    for arch in arch_list:
        n_epochs = 30
        print(arch)
        if not os.path.isfile(os.path.join(
                results_folder,
                f'{potential}_{arch}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}_final_{29}.npy')):

            net = getattr(models, arch)(pretrained=pretrained)
            n_params = count_weights_only(net)
            del net
            loader = get_loader(n_params, arch)
            dataset_length = len(loader)


            optimal_lr, exit_codes[arch], final_loss, final_acc = find_lr(arch, loader, dataset_length, n_epochs)

            for i in range(4):
                if exit_codes[arch]:
                    n_epochs += 30
                    optimal_lr, exit_codes[arch], final_loss, final_acc = find_lr(arch, loader, dataset_length, n_epochs)
            if exit_codes[arch]:
                print(f'Binary search failed, final loss {final_loss}, final acc {final_acc}, optimal lr {optimal_lr}',
                      flush=True)

            for seed in range(n_seeds):
                if not os.path.isfile(os.path.join(
                        results_folder,
                        f'{potential}_{arch}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}_final_{seed}.npy')):
                    final_loss, accuracy, init_weights, final_weights = train(arch, optimal_lr, loader,
                                                                              dataset_length, n_epochs)
                    if accuracy < 100:
                        n_bad_accuracy[arch] += 1
                        if accuracy < 90:
                            n_very_bad_accuracy[arch] += 1
                    print(f'{arch} seed {seed} accuracy {accuracy}')
                    save_weights(init_weights, final_weights, arch, seed)

    print('RUN SUMMARY\n\n')
    for arch in arch_list:
        print(f'{arch}:\n\texit codes\t{exit_codes[arch]}\n\t< 100 acc'
              f'\t{n_bad_accuracy[arch]}\n\t<90 acc\t{n_very_bad_accuracy[arch]}')


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Finetuning')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    main()
