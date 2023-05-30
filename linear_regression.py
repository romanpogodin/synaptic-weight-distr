import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required
from typing import List, Optional
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy.special import erf
import torch.optim as optim
from collections import defaultdict
import os
from pathlib import Path
from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

import md_utils


Section('data', 'data related stuff').params(
    results_folder=Param(str, 'where to save the data', required=True)
)


Section('training', 'training related stuff').params(
    corr_scale=Param(float, 'corr_scale', required=True),
)


def print_val_accuracy(net, testloader, device='cuda:0'):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        with autocast():
            for data in testloader:
                inputs = data[0].to(device)
                labels = data[1].to(device)
                # calculate outputs by running images through the network
                outputs = net(inputs)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct // total} %')


def standard_normal_cdf(x):
    return (1 + erf(x / math.sqrt(2))) / 2


def train_net(net, n_epochs, inputs, labels, optimizer, scheduler, criterion):
    net.train()

    running_loss = 0
    for epoch in range(n_epochs):
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            outputs = net(inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # small model, not necessary
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        running_loss = loss.item()
        scheduler.step()

    return running_loss


def train(inputs, labels, lr, n_epochs, potential):
    D = inputs.shape[1]
    net = nn.Linear(D, 1, bias=False)
    nn.init.normal_(net.weight, mean=0, std=1 / math.sqrt(D))
    net.to('cuda:0')
    net.train()

    init_weights = net.weight.detach().cpu().clone()

    optimizer = md_utils.SGD(net.parameters(), lr=lr, momentum=0.9, potential=potential, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.MSELoss()

    final_loss = train_net(net, n_epochs, inputs, labels, optimizer, scheduler, criterion)

    final_weights = net.weight.detach().cpu().clone()

    return final_loss, init_weights, final_weights


def find_lr(inputs, labels, n_epochs, potential):
    lr_range = np.logspace(-7, -1, 16)
    losses = np.zeros_like(lr_range)
    for i, lr in enumerate(lr_range):
        losses[i] = train(inputs, labels, lr_range[i], n_epochs, potential)[0]
    idx_argmin = np.nanargmin(losses)
    exit_code = 0 if losses[idx_argmin] < 1e-3 else 1
    return lr_range[idx_argmin], exit_code, losses[idx_argmin]


@param('data.results_folder')
@param('training.corr_scale')
def run_seeds(n_seeds, n_points, N_max, D_max, L, D_range, N_range, potential, points_for_normal, normal_cdf,
              n_d_dependency, corr_power, results_folder, corr_scale):
    exit_codes = dict()
    n_bad_loss = defaultdict(int)
    n_very_bad_loss = defaultdict(int)
    change_magnitude = np.zeros((n_seeds, n_points))
    density_mse = np.zeros((n_seeds, n_points))
    density_mse_uncentered = np.zeros((n_seeds, n_points))

    inputs = torch.randn(n_seeds + 1, N_max, D_max, device='cuda:0') @ L.T
    inputs += torch.rand(n_seeds + 1, N_max, D_max, device='cuda:0') - 0.5
    labels = 2 * torch.randint(0, 2, size=(n_seeds + 1, N_max, 1), dtype=torch.float32, device='cuda:0') - 1

    for D_idx, D in enumerate(D_range):
        N = N_range[D_idx]
        n_epochs = 100
        print(f'width {D}', flush=True)

        optimal_lr, exit_codes[f'{D}'], final_loss = find_lr(inputs[-1, :N, :D], labels[-1, :N], n_epochs,
                                                             potential.name)

        for i in range(4):
            if exit_codes[f'{D}']:
                n_epochs += 100
                optimal_lr, exit_codes[f'{D}'], final_loss = find_lr(inputs[-1, :N, :D], labels[-1, :N], n_epochs,
                                                                     potential.name)
        if exit_codes[f'{D}']:
            print(f'Binary search failed, final loss {final_loss}, optimal lr {optimal_lr}')

        for seed in range(n_seeds):
            final_loss, init_weights, final_weights = train(inputs[seed, :N, :D], labels[seed, :N], optimal_lr,
                                                            n_epochs, potential.name)
            if final_loss > 1e-3:
                n_bad_loss[f'{D}'] += 1
                if final_loss > 1e-2:
                    n_very_bad_loss[f'{D}'] += 1

            diff = potential.grad(final_weights.flatten()) - potential.grad(init_weights.flatten())
            init_norm = torch.norm(potential.grad(init_weights.flatten()))
            change_magnitude[seed, D_idx] = torch.norm(diff) / \
                                            (init_norm + 1e-10 * (init_norm < 1e-10))

            diff_norm = ((diff - diff.mean()) / diff.std()).numpy()
            empirical_cdf = (points_for_normal[:, None] > diff_norm[None, :]).mean(axis=-1)

            density_mse[seed, D_idx] = np.abs(empirical_cdf - normal_cdf).sum() * (
                        points_for_normal[1] - points_for_normal[0])

            diff_norm = (diff / diff.std()).numpy()
            empirical_cdf = (points_for_normal[:, None] > diff_norm[None, :]).mean(axis=-1)

            density_mse_uncentered[seed, D_idx] = np.abs(empirical_cdf - normal_cdf).sum() * (
                    points_for_normal[1] - points_for_normal[0])

    np.save(os.path.join(results_folder,
                         f'{potential.name}_nd{n_d_dependency:.2f}_corr{corr_power:.2f}_corr_scale{corr_scale}_magnitude'),
            change_magnitude)
    np.save(os.path.join(results_folder,
                         f'{potential.name}_nd{n_d_dependency:.2f}_corr{corr_power:.2f}_corr_scale{corr_scale}_cdf'),
            density_mse)
    np.save(os.path.join(results_folder,
                         f'{potential.name}_nd{n_d_dependency:.2f}_corr{corr_power:.2f}_corr_scale{corr_scale}_cdf_uncentered'),
            density_mse_uncentered)
    print('RUN SUMMARY\n\n')
    for D in D_range:
        exit_code = exit_codes[f'{D}']
        b_loss = n_bad_loss[f'{D}']
        vb_loss = n_very_bad_loss[f'{D}']
        print(f'{D}:\n\texit codes\t{exit_code}\n\t> 1e-3 loss'
              f'\t{b_loss}\n\t> 1e-2 loss\t{vb_loss}', flush=True)


def get_potential(potential):
    if potential == 'negative_entropy':
        return md_utils.NegativeEntropy()
    elif 'norm' in potential:
        return md_utils.Pnorm(float(potential.split('-norm')[0]))
    else:
        raise NotImplementedError('Potential {} is not implemented'.format(potential))


@param('training.corr_scale')
def main(corr_scale):
    for corr_power in [2, 1]:
        n_seeds = 30
        D_max = int(2e4)
        dist = 1 / (1 + np.abs(np.arange(-D_max, D_max + 1) / corr_scale) ** corr_power)
        dist = dist * np.power(-np.ones(2 * D_max + 1), np.arange(-D_max, D_max + 1))

        n_points = 10
        D_range = np.logspace(2, np.log10(D_max), n_points, dtype=int)

        points_for_normal = np.linspace(-5, 5, int(1e3))
        normal_cdf = standard_normal_cdf(points_for_normal)

        corr_matrix = np.zeros((D_max, D_max))
        for i in range(D_max):
            corr_matrix[i] = dist[D_max - i:D_max - i + D_max]
        L = torch.tensor(np.linalg.cholesky(corr_matrix), device='cuda:0', dtype=torch.float32)  # corr = L Lt

        for potential in ['2-norm', 'negative_entropy', '3-norm']:
            potential_func = get_potential(potential)
            for n_d_dependency in [0.75, 0.5]:
                N_range = np.power(D_range, n_d_dependency).astype(int)
                N_max = N_range[-1]

                print(f'{potential} n_d {n_d_dependency} corr {corr_power}', flush=True)
                run_seeds(n_seeds, n_points, N_max, D_max, L, D_range, N_range, potential_func, points_for_normal,
                          normal_cdf, n_d_dependency, corr_power)
        del L
        torch.cuda.empty_cache()


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Regression')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    if not quiet:
        config.summary()


if __name__ == "__main__":
    make_config()
    main()
