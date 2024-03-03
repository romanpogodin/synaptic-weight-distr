import numpy as np
import math
from scipy.special import erf
import md_utils
import torch

import os
from pathlib import Path
from argparse import ArgumentParser
from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf


Section('data', 'data related stuff').params(
    data_path=Param(str, 'Where the data is', required=True),
    results_folder=Param(str, 'where to save the data', required=True),
    pretrained=Param(bool, 'pretrained net or not', required=True)
)


def standard_normal_cdf(x):
    return (1 + erf(x / math.sqrt(2))) / 2


def get_flattened_weights(weights_dict, no_fc=False, no_downsample=False):
    weights = None
    for key, val in weights_dict.items():
        if ('fc' not in key or not no_fc) and ('downsample' not in key or not no_downsample):
            if weights is None:
                weights = val.flatten()
            else:
                weights = torch.cat((weights, val.flatten()))
    return weights


@param('data.data_path')
@param('data.pretrained')
def find_ratio_cdf_mse(n_seeds, arch_list, potential, other_potential, n_d_dependency, xent_penalty,
                       no_fc, no_downsample, data_path, pretrained):
    points_for_normal = np.linspace(-5, 5, int(1e3))
    normal_cdf = torch.tensor(standard_normal_cdf(points_for_normal), device='cuda:0')
    points_for_normal = torch.tensor(points_for_normal, device='cuda:0')

    n_points = len(arch_list)
    D_range = np.zeros(n_points)

    change_magnitude = np.zeros((n_seeds, n_points))
    density_mse = np.zeros((n_seeds, n_points))

    for idx, arch in enumerate(arch_list):
        filename_prefix = f'{potential.name}_{arch}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}'

        for seed in range(n_seeds):
            if seed == 0 or not pretrained:
                init_weights = get_flattened_weights(np.load(
                    os.path.join(data_path, filename_prefix + f'_init_{seed}.npy'), allow_pickle=True).item(),
                                                     no_fc, no_downsample).to('cuda:0')
                D_range[idx] = len(init_weights)

            final_weights = get_flattened_weights(np.load(
                os.path.join(data_path, filename_prefix + f'_final_{seed}.npy'), allow_pickle=True).item(),
                                                  no_fc, no_downsample).to('cuda:0')

            diff = other_potential.grad(final_weights) - other_potential.grad(init_weights)
            change_magnitude[seed, idx] = (torch.norm(diff) / torch.norm(other_potential.grad(init_weights))).item()

            diff = diff / diff.std() # ((diff - diff.mean()) / diff.std())#.numpy()

            empirical_cdf = torch.zeros_like(points_for_normal)

            # chunk_size = 1000 if arch not in ['resnet34', 'resnet50'] else 500
            # for i in range(len(points_for_normal) // chunk_size):
            #     empirical_cdf[chunk_size * i:chunk_size * (i + 1)] = \
            #         (points_for_normal[chunk_size * i:chunk_size * (i + 1), None] > diff[None, :]).mean(axis=-1)

            # split in half
            # empirical_cdf = (points_for_normal[:, None] > diff[None, :]).float().mean(axis=-1)
            chunk_size = 50#200
            for i in range(len(points_for_normal) // chunk_size):
                empirical_cdf[chunk_size * i:chunk_size * (i + 1)] = \
                    (points_for_normal[chunk_size * i:chunk_size * (i + 1), None] > diff[None, :]).float().mean(axis=-1)

            density_mse[seed, idx] = (torch.abs(empirical_cdf - normal_cdf).sum() *
                                     (points_for_normal[1] - points_for_normal[0])).item()
    return change_magnitude, density_mse, D_range


@param('data.results_folder')
def main(results_folder):
    n_seeds = 10
    no_fc = True
    no_downsample = False

    potential_list = [md_utils.Pnorm(2), md_utils.NegativeEntropy(), md_utils.Pnorm(3)]
    other_potential_list = [md_utils.Pnorm(2), md_utils.NegativeEntropy(), md_utils.Pnorm(3),
                            md_utils.Pnorm(1.5), md_utils.Pnorm(2.5)]
    arch_list = [100, 1000]
    n_d_list = [0.5]#, 0.75]
    # xent_penalty_list = [0]#, 0.01]
    xent_penalty = 0

    for potential in potential_list:
        for n_d_dependency in n_d_list:
            # for xent_penalty in xent_penalty_list:
            # other_potential = potential
            for other_potential in other_potential_list:
                print(f'{potential.name}, {n_d_dependency}, {xent_penalty} calculated for {other_potential.name}',
                      flush=True)
                change_magnitude, density_mse, D_range = find_ratio_cdf_mse(n_seeds, arch_list,
                                                                            potential, other_potential,
                                                                            n_d_dependency,
                                                                            xent_penalty,
                                                                            no_fc, no_downsample)
                filename_prefix = f'{potential.name}_nd{n_d_dependency:.2f}_xent{xent_penalty:.2f}'
                np.save(os.path.join(results_folder, filename_prefix + f'_change_magnitude_{other_potential.name}'),
                        change_magnitude)
                np.save(os.path.join(results_folder, filename_prefix + f'_density_mse_{other_potential.name}'),
                        density_mse)
                np.save(os.path.join(results_folder, filename_prefix + f'_D_range'), D_range)



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
