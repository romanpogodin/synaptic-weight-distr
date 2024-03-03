import numpy as np
import math
from scipy.special import erf
import md_utils
import torch
import pandas as pd
import os
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.path import Path

def finetuning_plots(results_folder, name_suffix=''):
    w = 1
    h = 0.2
    verts = [
        (-w, -h),  # left, bottom
        (-w, h),  # left, top
        (w, h),  # right, top
        (w, -h),  # right, bottom
        (-w, -h),  # back to left, bottom
    ]

    codes = [
        Path.MOVETO,  # begin drawing
        Path.LINETO,  # straight line
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,  # close shape. This is not required for this shape but is "good form"
    ]

    path = Path(verts, codes)
    potentials_name = ['negative entropy', '2-norm', '3-norm']

    def get_color(arch):
        if 'efficient' in arch:
            return '#AF3B3B'
        if 'shuffle' in arch:
            return '#56B4E9'
        return '#FD935E'

    for nd  in ['0.50']:
        fig, axes = plt.subplots(ncols=3, figsize=(15, 4), sharey=True, sharex=True)
        for idx, potential in enumerate(['negative_entropy', '2-norm', '3-norm']):
            other_potential = potential  # '2-norm'#potential#'negative_entropy'#'3-norm'
            mag_path = f'{potential}_nd{nd}_xent0.00_change_magnitude_{other_potential}.npy'
            cdf_path = f'{potential}_nd{nd}_xent0.00_density_mse_{other_potential}.npy'
            range_path = f'{potential}_nd{nd}_xent0.00_D_range.npy'

            #     magnitude_data = np.load(results_folder + mag_path)
            cdf_data = np.load(results_folder + cdf_path)
            range_data = np.load(results_folder + range_path)
            # same order as in data processing
            arch_list = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3',
                         'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'resnet18', 'resnet34', 'cornet-s', 'cornet-rt']
            if 'ssl' in name_suffix:
                arch_list = ['resnet50']
            arch_name = ['b0', 'b1', 'b2', 'b3', 'x0.5', 'x1', 'r18', 'r34', 'c-s', 'c-rt']
            for arch_idx, arch in enumerate(arch_list):
                axes[idx].scatter(range_data[arch_idx] * np.ones(cdf_data.shape[0]),
                                  cdf_data[:, arch_idx], edgecolors=get_color(arch), facecolors='none',
                                  s=200, alpha=0.5)
                axes[idx].scatter(range_data[arch_idx], cdf_data[:, arch_idx].mean(),
                                  marker=path, s=500, lw=2, facecolors=get_color(arch), edgecolors='k')
                axes[idx].set_xscale('log')
                axes[idx].set_xlabel('Number of weights')
                axes[idx].set_ylim(0, 1.05 * cdf_data.max())
                axes[idx].set_xlim(0.75 * range_data.min(), 1.5 * range_data.max())
                axes[idx].set_title(potentials_name[idx])
        axes[0].set_ylabel(r'$\Delta\mathrm{CDF}$')

        handles = [mpatches.Patch(color=get_color('efficient'), label='EfficientNet'),
                   mpatches.Patch(color=get_color('shuffle'), label='ShuffleNetv2'),
                   mpatches.Patch(color=get_color('resnet'), label='ResNet')]
        # axes[0].legend(handles=handles)
        axes[0].set_yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])


        plt.tight_layout()
        plt.savefig(f'finetuning_plots_uncentered_{nd}{name_suffix}.pdf')
        plt.show()


def finetuning_plots_rnn(results_folder):
    w = 1
    h = 0.2
    verts = [
        (-w, -h),  # left, bottom
        (-w, h),  # left, top
        (w, h),  # right, top
        (w, -h),  # right, bottom
        (-w, -h),  # back to left, bottom
    ]

    codes = [
        Path.MOVETO,  # begin drawing
        Path.LINETO,  # straight line
        Path.LINETO,
        Path.LINETO,
        Path.CLOSEPOLY,  # close shape. This is not required for this shape but is "good form"
    ]

    path = Path(verts, codes)
    potentials_name = ['negative entropy', '2-norm', '3-norm']

    def get_color(arch):
        # if 'efficient' in arch:
        #     return '#AF3B3B'
        # if 'shuffle' in arch:
        #     return '#56B4E9'
        return '#FD935E'

    for nd  in ['0.50']:
        fig, axes = plt.subplots(ncols=3, figsize=(15, 4), #sharey=True,
                                 sharex=True)
        for idx, potential in enumerate(['negative_entropy', '2-norm', '3-norm']):
            other_potential = potential  # '2-norm'#potential#'negative_entropy'#'3-norm'
            mag_path = f'{potential}_nd{nd}_xent0.00_change_magnitude_{other_potential}.npy'
            cdf_path = f'{potential}_nd{nd}_xent0.00_density_mse_{other_potential}.npy'
            range_path = f'{potential}_nd{nd}_xent0.00_D_range.npy'

            #     magnitude_data = np.load(results_folder + mag_path)
            cdf_data = np.load(results_folder + cdf_path)
            range_data = np.load(results_folder + range_path)
            # same order as in data processing
            arch_list = [100, 1000]
            arch_name = [100, 1000]
            for arch_idx, arch in enumerate(arch_list):
                axes[idx].scatter(range_data[arch_idx] * np.ones(cdf_data.shape[0]),
                                  cdf_data[:, arch_idx], edgecolors=get_color(arch), facecolors='none',
                                  s=200, alpha=0.5)
                axes[idx].scatter(range_data[arch_idx], cdf_data[:, arch_idx].mean(),
                                  marker=path, s=500, lw=2, facecolors=get_color(arch), edgecolors='k')
                axes[idx].set_xscale('log')
                axes[idx].set_xlabel('Number of weights')
                axes[idx].set_ylim(0, 1.05 * cdf_data.max())
                axes[idx].set_xlim(0.75 * range_data.min(), 1.5 * range_data.max())
                axes[idx].set_title(potentials_name[idx])
        axes[0].set_ylabel(r'$\Delta\mathrm{CDF}$')

        handles = [mpatches.Patch(color=get_color('efficient'), label='EfficientNet'),
                   mpatches.Patch(color=get_color('shuffle'), label='ShuffleNetv2'),
                   mpatches.Patch(color=get_color('resnet'), label='ResNet')]
        # axes[0].legend(handles=handles)

        # axes[0].set_yticks([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])

        plt.tight_layout()
        plt.savefig(f'finetuning_plots_uncentered_{nd}_rnn.pdf')
        plt.show()



def finetuning_plots_phi(results_folder, name_suffix=''):
    for nd in ['0.50']:
        arch_list = ['EfficientNet b0', 'EfficientNet b1', 'EfficientNet b2', 'EfficientNet b3',
                     'ShuffleNet v2 x0.5', 'ShuffleNet v2 x1', 'ResNet18', 'ResNet34', 'CORNet-S', 'CORNet-RT']
        if 'ssl' in name_suffix:
            arch_list = ['resnet50']
        arch_idx_list = np.arange(len(arch_list))

        fig, axes = plt.subplots(ncols=max(1, len(arch_idx_list) // 2),
                                 nrows=min(2, len(arch_list)),
                                 figsize=(8 if len(arch_list) == 1 else 15, 6), sharey=True, sharex=True)
        if len(arch_list) > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        colors = ['#0072B2', '#009E73', '#D55E00']
        potentials = ['negative_entropy', '2-norm', '3-norm']
        other_potentials = ['negative_entropy', '1.5-norm', '2-norm', '2.5-norm', '3-norm']
        potentials_name = ['negative entropy (NE)', '2-norm', '3-norm']
        potentials_name2 = ['NE', '2-norm', '3-norm']
        other_potentials_names = ['NE', '1.5', '2', '2.5', '3']
        for idx, arch_idx in enumerate(arch_idx_list):
            for p_idx, potential in enumerate(potentials):
                data = np.zeros((len(other_potentials), 10))
                for op_idx, other_potential in enumerate(other_potentials):
                    cdf_path = f'{potential}_nd{nd}_xent0.00_density_mse_{other_potential}.npy'
                    data[op_idx, :] = np.load(results_folder + cdf_path)[:, arch_idx]

                axes[idx].plot(data.mean(axis=1), color=colors[p_idx], label=f'{potentials_name2[p_idx]}',
                               lw=4)
                axes[idx].scatter(np.arange(len(other_potentials)).repeat(10),
                                  data.flatten(), edgecolors=colors[p_idx], facecolors='none',
                                  s=200, alpha=0.5)
            axes[idx].set_title(arch_list[arch_idx])
            axes[idx].set_xticks(np.arange(len(other_potentials_names)), other_potentials_names)
        # axes[0].legend()
        axes[0].set_ylabel(r'$\Delta\mathrm{CDF}$')
        axes[4].set_ylabel(r'$\Delta\mathrm{CDF}$')

        axes[0].set_yticks([0.0, 0.2, 0.4, 0.6])

        plt.tight_layout()
        plt.savefig(f'finetuning_diff_phi_{nd}{name_suffix}.pdf')
        plt.show()


def finetuning_plots_phi_rnn(results_folder):
    for nd in ['0.50']:

        arch_list = [100, 1000]
        arch_idx_list = np.arange(len(arch_list))
        fig, axes = plt.subplots(ncols=2, nrows=len(arch_idx_list) // 2, figsize=(15, 6),# sharey=True,
                                 sharex=True)
        axes = axes.flatten()

        colors = ['#0072B2', '#009E73', '#D55E00']
        potentials = ['negative_entropy', '2-norm', '3-norm']
        other_potentials = ['negative_entropy', '1.5-norm', '2-norm', '2.5-norm', '3-norm']
        potentials_name = ['negative entropy (NE)', '2-norm', '3-norm']
        potentials_name2 = ['NE', '2-norm', '3-norm']
        other_potentials_names = ['NE', '1.5', '2', '2.5', '3']
        for idx, arch_idx in enumerate(arch_idx_list):
            for p_idx, potential in enumerate(potentials):
                data = np.zeros((len(other_potentials), 10))
                for op_idx, other_potential in enumerate(other_potentials):
                    cdf_path = f'{potential}_nd{nd}_xent0.00_density_mse_{other_potential}.npy'
                    data[op_idx, :] = np.load(results_folder + cdf_path)[:, arch_idx]

                axes[idx].plot(data.mean(axis=1), color=colors[p_idx], label=f'{potentials_name2[p_idx]}',
                               lw=4)
                axes[idx].scatter(np.arange(len(other_potentials)).repeat(10),
                                  data.flatten(), edgecolors=colors[p_idx], facecolors='none',
                                  s=200, alpha=0.5)
            axes[idx].set_title(arch_list[arch_idx])
            axes[idx].set_xticks(np.arange(len(other_potentials_names)), other_potentials_names)
        # axes[0].legend()
        axes[0].set_ylabel(r'$\Delta\mathrm{CDF}$')
        # axes[4].set_ylabel(r'$\Delta\mathrm{CDF}$')

        # axes[0].set_yticks([0.0, 0.2, 0.4, 0.6])

        plt.tight_layout()
        plt.savefig(f'finetuning_diff_phi_{nd}_rnn.pdf')
        plt.show()


def linear_regression_plots(folder):
    corr_scale = 1
    D_range = np.logspace(2, np.log10(2e4), 10, dtype=int)

    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 8), sharex=True)

    colors = ['#0072B2', '#D55E00']

    for c_idx, corr_power in enumerate([2, 1]):
        for r_idx, potential in enumerate(['negative_entropy', '2.0-norm', '3.0-norm']):  # , '1.5-norm', '2.5-norm']):
            for nd_idx, n_d_dependency in enumerate([0.75, 0.5]):
                stat = 'cdf_uncentered'
                data = np.load(
                    folder + f'{potential}_nd{n_d_dependency:.2f}_corr{corr_power:.2f}_corr_scale{corr_scale:.1f}_{stat}.npy')

                axes[r_idx, 2 * c_idx].plot(D_range, np.quantile(data, 0.5, axis=0), lw=3,
                                            label=fr'$N=D^{{{n_d_dependency}}}$', color=colors[nd_idx])
                axes[r_idx, 2 * c_idx].fill_between(D_range,
                                                    np.quantile(data, 0.05, axis=0),
                                                    np.quantile(data, 0.95, axis=0), alpha=0.3, color=colors[nd_idx])
                axes[r_idx, 2 * c_idx].set_title(f'{potential}, p={corr_power}')
                #             axes[r_idx, 2 * c_idx].legend()
                axes[r_idx, 2 * c_idx].set_ylim(bottom=0, top=0.2)
                axes[r_idx, 2 * c_idx].set_xscale('log')

                stat = 'magnitude'
                data = np.load(
                    folder + f'{potential}_nd{n_d_dependency:.2f}_corr{corr_power:.2f}_corr_scale{corr_scale:.1f}_{stat}.npy')
                #             if c_idx == 1 and r_idx == 0:
                #                 print(data)
                axes[r_idx, 2 * c_idx + 1].plot(D_range, np.nanquantile(data, 0.5, axis=0), lw=3,
                                                label=fr'$N=D^{{{n_d_dependency}}}$', color=colors[nd_idx])
                axes[r_idx, 2 * c_idx + 1].fill_between(D_range,
                                                        np.nanquantile(data, 0.05, axis=0),
                                                        np.nanquantile(data, 0.95, axis=0), alpha=0.3,
                                                        color=colors[nd_idx])
                axes[r_idx, 2 * c_idx + 1].set_title(f'{potential}, p={corr_power}')
                axes[r_idx, 2 * c_idx + 1].set_ylim(bottom=0, top=1)
                axes[r_idx, 2 * c_idx + 1].set_ylim(bottom=0)  # , top=0.2)
                axes[r_idx, 2 * c_idx + 1].set_xscale('log')
    axes[0, 0].legend()
    plt.tight_layout()
    plt.savefig('linreg.pdf')
    plt.show()


def robustness_plots():
    def standard_normal_cdf(x):
        return (1 + erf(x / math.sqrt(2))) / 2

    points_for_normal = np.linspace(-5, 5, int(1e3))
    normal_cdf = standard_normal_cdf(points_for_normal)

    def lognormal_init(D):
        std = 1 / math.sqrt(D)

        def calc_ln_mu_sigma(mean, var):
            "Given desired mean and var returns ln mu and sigma"
            mu_ln = math.log(mean ** 2 / math.sqrt(mean ** 2 + var))
            sigma_ln = math.sqrt(math.log(1 + (var / mean ** 2)))
            return mu_ln, sigma_ln

        mu, sigma = calc_ln_mu_sigma(std, std ** 2)

        w = torch.zeros(D)
        w.log_normal_(mu, sigma)
        w.mul_(2 * torch.bernoulli(0.5 * torch.ones_like(w)) - 1)
        return w

    D = 1e2
    n_seeds = 30
    D_sample = 1000

    potential_true_list = [md_utils.NegativeEntropy(), md_utils.Pnorm(2), md_utils.Pnorm(3)]
    potential_list = [md_utils.NegativeEntropy()] + [md_utils.Pnorm(i) for i in np.linspace(1, 3, 21)[1:]]
    potential_names = ['NE'] + [f'{i:.1f}' for i in np.linspace(1, 3, 21)[1:]]

    change_magnitude = torch.zeros((len(potential_true_list), 30, len(potential_list)))
    density_mse = torch.zeros((len(potential_true_list), 30, len(potential_list)))
    density_mse_uncentered = torch.zeros((len(potential_true_list), 30, len(potential_list)))

    for pt_idx, potential_true in enumerate(potential_true_list):
        for seed in range(n_seeds):
            w0 = lognormal_init(D_sample)  # torch.randn(D_sample) / np.sqrt(D)
            phi_w0_scale = potential_true.grad(torch.tensor(1 / np.sqrt(D)))
            phi_w0 = potential_true.grad(w0)
            phi_w = phi_w0 + 0.1 * phi_w0_scale * torch.randn(D_sample)
            w = potential_true.inv_grad(phi_w, sign_w=torch.sign(w0))
            for p_idx, potential in enumerate(potential_list):
                diff = potential.grad(w) - potential.grad(w0)
                init_norm = torch.norm(potential.grad(w0))
                change_magnitude[pt_idx, seed, p_idx] = torch.norm(diff) / \
                                                        (init_norm + 1e-10 * (init_norm < 1e-10))

                diff_norm = ((diff - diff.mean()) / diff.std()).numpy()
                empirical_cdf = (points_for_normal[:, None] > diff_norm[None, :]).mean(axis=-1)

                density_mse[pt_idx, seed, p_idx] = np.abs(empirical_cdf - normal_cdf).sum() * (
                        points_for_normal[1] - points_for_normal[0])

                diff_norm = (diff / diff.std()).numpy()
                empirical_cdf = (points_for_normal[:, None] > diff_norm[None, :]).mean(axis=-1)

                density_mse_uncentered[pt_idx, seed, p_idx] = np.abs(empirical_cdf - normal_cdf).sum() * (
                        points_for_normal[1] - points_for_normal[0])

    plt.figure(figsize=(5, 5))

    mean = density_mse_uncentered.mean(dim=1)
    std = density_mse_uncentered.std(dim=1)

    colors = ['#0072B2', '#009E73', '#D55E00']

    for pt_idx, potential in enumerate(potential_true_list):
        plt.plot(mean[pt_idx], label=potential.name, color=colors[pt_idx])
        plt.fill_between(np.arange(len(potential_list)), mean[pt_idx] - std[pt_idx], mean[pt_idx] + std[pt_idx],
                         alpha=0.5, color=colors[pt_idx])
    plt.xticks(np.arange(len(potential_list)), potential_names)
    plt.title('lognormal initialization')
    plt.tight_layout()
    plt.savefig('robustness_lognormal.pdf')
    plt.show()

    D = 1e2
    n_seeds = 30
    D_sample = 1000

    potential_true_list = [md_utils.NegativeEntropy(), md_utils.Pnorm(2), md_utils.Pnorm(3)]
    potential_list = [md_utils.NegativeEntropy()] + [md_utils.Pnorm(i) for i in np.linspace(1, 3, 21)[1:]]
    potential_names = ['NE'] + [f'{i:.1f}' for i in np.linspace(1, 3, 21)[1:]]

    change_magnitude = torch.zeros((len(potential_true_list), 30, len(potential_list)))
    density_mse = torch.zeros((len(potential_true_list), 30, len(potential_list)))
    density_mse_uncentered = torch.zeros((len(potential_true_list), 30, len(potential_list)))

    for pt_idx, potential_true in enumerate(potential_true_list):
        for seed in range(n_seeds):
            w0 = torch.randn(D_sample) / np.sqrt(D)
            phi_w0_scale = potential_true.grad(torch.tensor(1 / np.sqrt(D)))
            phi_w0 = potential_true.grad(w0)
            phi_w = phi_w0 + 0.1 * phi_w0_scale * torch.randn(D_sample)
            w = potential_true.inv_grad(phi_w, sign_w=torch.sign(w0))
            for p_idx, potential in enumerate(potential_list):
                diff = potential.grad(w) - potential.grad(w0)
                init_norm = torch.norm(potential.grad(w0))
                change_magnitude[pt_idx, seed, p_idx] = torch.norm(diff) / \
                                                        (init_norm + 1e-10 * (init_norm < 1e-10))

                diff_norm = ((diff - diff.mean()) / diff.std()).numpy()
                empirical_cdf = (points_for_normal[:, None] > diff_norm[None, :]).mean(axis=-1)

                density_mse[pt_idx, seed, p_idx] = np.abs(empirical_cdf - normal_cdf).sum() * (
                        points_for_normal[1] - points_for_normal[0])

                diff_norm = (diff / diff.std()).numpy()
                empirical_cdf = (points_for_normal[:, None] > diff_norm[None, :]).mean(axis=-1)

                density_mse_uncentered[pt_idx, seed, p_idx] = np.abs(empirical_cdf - normal_cdf).sum() * (
                        points_for_normal[1] - points_for_normal[0])

    plt.figure(figsize=(5, 5))

    mean = density_mse_uncentered.mean(dim=1)
    std = density_mse_uncentered.std(dim=1)

    colors = ['#0072B2', '#009E73', '#D55E00']

    for pt_idx, potential in enumerate(potential_true_list):
        plt.plot(mean[pt_idx], label=potential.name, color=colors[pt_idx])
        plt.fill_between(np.arange(len(potential_list)), mean[pt_idx] - std[pt_idx], mean[pt_idx] + std[pt_idx],
                         alpha=0.5, color=colors[pt_idx])
    plt.xticks(np.arange(len(potential_list)), potential_names)
    plt.title('normal initialization')
    plt.tight_layout()
    plt.savefig('robustness_normal.pdf')
    plt.show()


def main():
    sns.set(font_scale=1.5, style='ticks')
    finetuning_plots('finetuning_results/')
    finetuning_plots_phi('finetuning_results/')
    finetuning_plots('finetuning_results_ssl/', name_suffix='_ssl')
    finetuning_plots_phi('finetuning_results_ssl/', name_suffix='_ssl')
    finetuning_plots_rnn('rnn_results/')
    finetuning_plots_phi_rnn('rnn_results/')
    linear_regression_plots('linreg_results/')
    robustness_plots()

if __name__ == '__main__':
    main()
