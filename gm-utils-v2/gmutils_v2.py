import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from box import Box as box
from datetime import datetime, timedelta
from functools import partial
from itertools import combinations
from natsort import natsorted
from pathlib import Path
from typing import Union, Optional

from matplotlib.figure import FigureBase
from matplotlib.gridspec import GridSpec

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA as ICA
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

from umap import UMAP


Model = Union[PCA, ICA, UMAP]


warnings.filterwarnings("ignore", category=ConvergenceWarning)
print(
    f"\nCurrent time: {datetime.date(datetime.now())} {datetime.time(datetime.now())}\n"
)


RANDOM_SEED = 6

WINDOW_STRIDE = 1

ALPHA = 0.4
TRAIN_TEST_COLORS = ["crimson", "lime", "dodgerblue"]

MSE = partial(mean_squared_error, multioutput="raw_values")

mean_shaped = partial(np.mean, axis=1, keepdims=True)
max_shaped = partial(np.max, axis=1, keepdims=True)
sort = partial(natsorted, key=lambda x: x[0])


E = "e"
F = "f"
D = "d"
P = "%"


def sf_str(val: float, sf: int = 5, num_type: str = F, sign: bool = False) -> float:
    if val == 0:
        return 0.0

    pre_decimal = 1 if num_type == E else int(np.log10(abs(val))) + 1

    post_decimal = sf - pre_decimal
    post_decimal = post_decimal + 1 if abs(val) < 1 and num_type != E else post_decimal

    if post_decimal < 0:
        val = round(val, post_decimal)
        post_decimal = 0

    return f"{float(val):{'+' if sign else ''}{pre_decimal}.{post_decimal}{num_type}}"


def normalize_signal(a):
    return (a - mean_shaped(a)) / max_shaped(np.abs(a))


def get_window_width(w, t, sig_length, factor=10, overlapped=True, stride=1):
    freq = w / 2 / np.pi
    n_waves_tot = freq * (t.max() - t.min())
    wave_width = t.size / n_waves_tot

    if n_waves_tot <= 2 * factor:
        return int(n_waves_tot / 2 * wave_width)

    window_width = int(factor * wave_width)
    if not overlapped:
        stride = window_width
    return (
        int(sig_length / 2) if sig_length <= 2 * stride + window_width else window_width
    )


def train_test_split(sig, ratios, warn=False):
    ratios_sum = sum(ratios)

    n_points_sig = sig.size
    n_points_unit = int(n_points_sig / ratios_sum)
    n_points_tot = n_points_unit * ratios_sum

    n_points_diff = n_points_sig - n_points_tot
    if warn and n_points_diff > 0:
        print(
            f"\nWARNING: Signal size not an exact match: the last "
            f"{n_points_diff} data points will be lost during splitting.\n"
        )

    idx_ls = []
    current_idx = 0
    for ratio in ratios:
        idx = np.arange(0, ratio * n_points_unit) + current_idx
        current_idx = idx.max() + 1
        idx_ls.append(idx)

    return [sig[idx] for idx in idx_ls], idx_ls


def window_stack(a, window_width, window_stride=WINDOW_STRIDE):
    # From https://stackoverflow.com/a/15722507
    if window_width == a.size:
        return a
    if window_width < window_stride:
        print(
            f"\nWARNING: Window width ({window_width}) is greater than window "
            f"stride ({window_stride}). This will result in some data being lost "
            "between samples.\n"
        )
    return np.vstack(
        [
            a[i : i - window_width + 1 or None : window_stride]
            for i in range(window_width)
        ]
    )


def process_signal(
    sigs: Union[np.ndarray, tuple[np.ndarray], list[np.ndarray]],
    window_width,
    window_stride=WINDOW_STRIDE,
    n_components=3,
    do_umap=False,
    umap_n_neighbors=None,
    umap_min_dist=0.1,
):
    sigs = sigs.copy()
    if isinstance(sigs, np.ndarray):
        sigs = [sigs]

    sig_train = sigs[0]

    sig_train_stacked = window_stack(sig_train, window_width, window_stride)

    scaler = StandardScaler()
    sig_train_scaled = scaler.fit_transform(sig_train_stacked)

    pca = PCA(n_components, random_state=RANDOM_SEED, tol=1e-10, whiten=True)
    ica = ICA(
        n_components,
        random_state=RANDOM_SEED,
        tol=5e-4,
        max_iter=500,
        whiten="unit-variance",
    )
    umap = None
    if do_umap:
        umap = UMAP(
            n_components=n_components,
            n_neighbors=umap_n_neighbors or sig_train_stacked.shape[0] - 1,
            random_state=RANDOM_SEED,
            min_dist=umap_min_dist,
        )

    [m.fit(sig_train_scaled) for m in (pca, ica, umap) if m]

    sigs_reshaped_ls = []
    pca_comps_ls = []
    ica_comps_ls = []
    umap_comps_ls = []
    for sig in sigs:
        sig_stacked = window_stack(sig, window_width, window_stride)
        sig_scaled = scaler.transform(sig_stacked)
        sigs_reshaped_ls.append(sig_scaled)
        pca_comps_ls.append(pca.transform(sig_scaled).T)
        ica_comps_ls.append(ica.transform(sig_scaled).T)
        if do_umap:
            umap_comps_ls.append(umap.transform(sig_scaled).T)

    return (
        (pca, pca_comps_ls, [pca.components_]),
        (ica, ica_comps_ls, [ica.mixing_.T]),
        (umap, umap_comps_ls),
        sigs_reshaped_ls,
    )


def plot_signal(
    t,
    comps_ls,
    sigs_idx_ls,
    colors,
    fig,
    gs,
    title=None,
    legend_vloc=-13.6,
):
    ax_sig = fig.add_subplot(gs)
    ax_sig.set_title(title, fontsize=16)

    n_sigs = len(comps_ls)
    if n_sigs > 1:
        plot_labels = ["Training Set", *[f"Testing Set {n}" for n in range(1, n_sigs)]]
        for x, sig, c, l in zip(sigs_idx_ls, comps_ls, colors, plot_labels):
            ax_sig.plot(t[x], sig, color=c, label=l)
        plt.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, legend_vloc),
            ncol=3,
            frameon=False,
            fontsize=14,
        )
    else:
        ax_sig.plot(t[sigs_idx_ls[0]], comps_ls[0], color="k")

    ax_sig.set_xlabel("Time [s]", fontsize=14)
    ax_sig.set_ylabel("g", fontsize=14)


def plot_components_1d(
    comps_ls, comp_str, colors, fig, gs, vecs=True, comps_range=None
):
    def plot_1d(comp_n, axs, cn, c, i=None, alpha=None):
        axs[cn].set_title(
            f"{comp_str}C{cn + 1}{'' if vecs else ' Scores'}", fontsize=15
        )
        axs[cn].plot(comp_n, c=c, alpha=alpha or 1)
        axs[cn].set_xlim(0, comp_n.size)

    ax_range = range(3)

    if not comps_range:
        comps_range = ax_range
    axs = [fig.add_subplot(gs[1 + i, 0 if comp_str == "P" else 1]) for i in ax_range]
    if len(comps_ls) == 1:
        c = "steelblue" if comp_str == "P" else "red"
        for cn, comp_n in zip(ax_range, comps_ls[0]):
            plot_1d(comp_n, axs, cn, c)
    else:
        for i, (comps, c) in enumerate(zip(comps_ls, colors)):
            for cn, comp_n in zip(ax_range, comps):
                plot_1d(comp_n, axs, cn, c, i, ALPHA)


def plot_components_2d(
    comps_ls, comp_str, colors, fig, gs, scatter=False, comps_range=None, vecs=False
):
    def plot_2d(axs, n, comps, comb, i=None, c=None, alpha=None):
        if not c:
            c = "r" if comp_str == "I" else "steelblue" if comp_str == "P" else "lime"

        comp1 = f"{comp_str}C{comb[0] + 1}"
        comp2 = f"{comp_str}C{comb[1] + 1}"

        axs[n].set_title(f"{comp1} vs {comp2}{'' if vecs else ' Scores'}", fontsize=14)
        axs[n].plot(
            comps[comb[0]],
            comps[comb[1]],
            "." if scatter else "-",
            c=c,
            alpha=alpha or (1 if not scatter else ALPHA),
            mew=0,
        )

        axs[n].set_aspect("equal", "datalim")
        axs[n].set_xlabel(comp1, fontsize=14)
        axs[n].set_ylabel(comp2, fontsize=14)

        axs[n].locator_params("x", nbins=3)
        axs[n].locator_params("y", nbins=3)

    ax_range = range(3)

    if not comps_range:
        comps_range = ax_range
    if comp_str == "UMAP":
        axs = [fig.add_subplot(gs[1, i]) for i in ax_range]
    else:
        axs = [fig.add_subplot(gs[1 if comp_str == "P" else 2, i]) for i in ax_range]
    if len(comps_ls) == 1:
        comps = comps_ls[0]
        for n, comb in enumerate(combinations(comps_range, 2)):
            plot_2d(axs, n, comps, comb)
    else:
        for i, (comps, c) in enumerate(zip(comps_ls, colors)):
            for n, comb in enumerate(combinations(comps_range, 2)):
                plot_2d(axs, n, comps, comb, i, c, ALPHA)


def plot_2_components_1d(comps_ls, comp_str, colors, fig, gs, vecs=True):
    def plot_1d(comp_n, axs, cn, c, i=None, alpha=None):
        axs[cn].set_title(
            f"{comp_str}C{cn + 1}{'' if vecs else ' Scores'}", fontsize=15
        )
        axs[cn].plot(
            comp_n,
            "." if comp_str == "UMAP" else "-",
            c=c,
            alpha=alpha or 1,
            mew=0,
        )
        axs[cn].set_xlim(0, comp_n.size)

    comps_range = range(len(comps_ls[0]))
    axs = [fig.add_subplot(gs[1 + i, 0]) for i in comps_range]
    if len(comps_ls) == 1:
        c = "r" if comp_str == "I" else "steelblue" if comp_str == "P" else "lime"
        for cn, comp_n in enumerate(comps_ls[0]):
            plot_1d(comp_n, axs, cn, c)
    else:
        for i, (comps, c) in enumerate(zip(comps_ls, colors)):
            for cn, comp_n in enumerate(comps):
                plot_1d(comp_n, axs, cn, c, i, ALPHA)


def plot_2_components_2d(
    comps_ls, comp_str, colors, fig, gs, vecs=False, scatter=False
):
    def plot_2d(ax, comps, comb, c=None, alpha=None):
        comp1 = f"{comp_str}C{comb[0] + 1}"
        comp2 = f"{comp_str}C{comb[1] + 1}"

        ax.set_title(f"{comp1} vs {comp2}{'' if vecs else ' Scores'}", fontsize=14)

        if not c:
            c = "r" if comp_str == "I" else "steelblue" if comp_str == "P" else "lime"

        ax.plot(
            comps[comb[0]],
            comps[comb[1]],
            "." if comp_str[0] == "U" or scatter else "-",
            c=c,
            alpha=alpha or (1 if not scatter or comp_str[0] == "U" else ALPHA),
            mew=0,
        )

        ax.set_aspect("equal", "datalim")
        ax.set_box_aspect(1)
        ax.set_xlabel(comp1, fontsize=14)
        ax.set_ylabel(comp2, fontsize=14)

        ax.locator_params("x", nbins=3)
        ax.locator_params("y", nbins=3)

    comps_range = range(len(comps_ls[0]))
    ax = fig.add_subplot(gs)
    if len(comps_ls) == 1:
        comps = comps_ls[0]
        for comb in combinations(comps_range, 2):
            plot_2d(ax, comps, comb)
    else:
        for comps, c in zip(comps_ls, colors):
            for comb in combinations(comps_range, 2):
                plot_2d(ax, comps, comb, c, ALPHA)


def get_reconstruction_error_one(
    model: Model,
    sig_reshaped: np.ndarray,
    comps_model: np.ndarray,
):
    sig_re = model.inverse_transform(comps_model.T)
    return MSE(sig_reshaped, sig_re)


def get_reconstruction_error_comps(
    model: Model,
    sigs_reshaped: list,
    comps_model: list,
):
    n_comps = len(comps_model)
    comps_range = np.arange(n_comps)

    re_ls = []
    for i in comps_range:
        comp_n = comps_model.copy()
        rows_0 = comps_range[np.arange(n_comps) != i]
        for row in rows_0:
            comp_n[row] = 0
        re_ls.append(get_reconstruction_error_one(model, sigs_reshaped, comp_n))

    return re_ls


def plot_reconstruction_error(
    model: Model,
    sig_reshaped_ls: np.ndarray,
    comps_model_ls: np.ndarray,
    comp_str: str,
    fig: FigureBase,
    gs: GridSpec,
    comps_range: range = None,
    legend_vloc: float = -1,
    legend_cols: int = 4,
):
    colors = TRAIN_TEST_COLORS
    n_sets = len(comps_model_ls)
    n_comps = len(comps_model_ls[0])

    plot_labels = ["Training Set", *[f"Testing Set {n}" for n in range(1, n_sets)]]

    max_features = np.max([sig_r.shape[1] for sig_r in sig_reshaped_ls])
    re_ls = []

    ax = fig.add_subplot(gs)
    ax.set_title(f"{comp_str}CA" if comp_str in {"P", "I"} else comp_str, fontsize=14)

    if not comps_range:
        comps_range = range(n_comps)

    if n_sets == 1:
        c = "r" if comp_str == "I" else "steelblue" if comp_str == "P" else "lime"
        re = get_reconstruction_error_one(model, sig_reshaped_ls[0], comps_model_ls[0])
        ax.plot(range(1, len(re) + 1), re, c="k", label="Model", alpha=ALPHA)
        re_ls.append(re)

        re_comps = get_reconstruction_error_comps(
            model, sig_reshaped_ls[0], comps_model_ls[0]
        )
        for n, c, re in zip(comps_range, plt.cm.Set1(range(n_comps)), re_comps):
            ax.plot(
                range(1, len(re) + 1), re, c=c, label=f"{comp_str}C{n + 1}", alpha=ALPHA
            )
        [re_ls.append(c) for c in re_comps]
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, legend_vloc),
            ncol=legend_cols,
            frameon=False,
            fontsize=14,
        )
    else:
        for n, (sig_r, comps, l, c) in enumerate(
            zip(sig_reshaped_ls, comps_model_ls, plot_labels, colors)
        ):
            re = get_reconstruction_error_one(model, sig_r, comps)
            ax.plot(
                re,
                c=c,
                label=l,
                alpha=ALPHA if n_sets != 1 else 1,
                zorder=6 - n,
            )
            re_ls.append(re)

    ax.set_xlabel("Features", fontsize=14)
    ax.set_ylabel("Reconstruction\nError (MSE)", fontsize=14)
    ax.set_xlim(1, max_features)
    ylim_pad = np.max(re_ls) * 0.1
    ax.set_ylim(-ylim_pad, 11 * ylim_pad)


def generate_results(
    t,
    sig,
    n_waves,
    w,
    title,
    fpath,
    fsuffix,
    train_test_ratios=(1, 1, 1),
    scatter_comps=True,
    do_umap=False,
    skip_n=None
):
    fpath.mkdir(exist_ok=True, parents=True)
    (fpath / "pdf").mkdir(exist_ok=True, parents=True)

    sigs, sigs_idx_ls = train_test_split(sig, train_test_ratios)

    sig_length = min(len(sig) for sig in sigs)
    window_width = get_window_width(w, t, sig_length, factor=n_waves)
    if any([i in fpath.parts for i in ["LHS", "constant", "sweep"]]):
        window_width = 500
    if window_width >= sig_length - 2:
        window_width //= 2

    ################
    # 2 COMPONENTS #
    ################

    if skip_n != 2:
        n_components = 2

        if window_width > n_components:
            t_now = time.perf_counter()
            pca_, ica_, umap_, sigs_reshaped_ls = process_signal(
                sigs,
                window_width=window_width,
                n_components=n_components,
            )

            pca, pca_comps_ls, pca_vecs_ls = pca_
            ica, ica_comps_ls, ica_vecs_ls = ica_
            umap, umap_comps_ls = umap_

            for model, vecs_ls, comps_ls, comp_str in zip(
                [pca, ica],
                [pca_vecs_ls, ica_vecs_ls],
                [pca_comps_ls, ica_comps_ls],
                ["P", "I"],
            ):
                fig = plt.figure(figsize=(10, 6.5))
                gs = GridSpec(
                    figure=fig,
                    nrows=4,
                    ncols=2,
                    height_ratios=[0.5, 1, 1, 1],
                    width_ratios=[1.5, 1],
                )

                plot_signal(
                    t,
                    sigs,
                    sigs_idx_ls,
                    title=title,
                    colors=TRAIN_TEST_COLORS,
                    fig=fig,
                    gs=gs[0, :],
                )
                plot_2_components_1d(
                    vecs_ls,
                    comp_str,
                    TRAIN_TEST_COLORS,
                    fig,
                    gs,
                    vecs=True,
                )
                plot_2_components_2d(
                    vecs_ls,
                    comp_str,
                    TRAIN_TEST_COLORS,
                    fig,
                    gs[1:3, 1],
                    vecs=True,
                    scatter=scatter_comps,
                )
                plot_reconstruction_error(
                    model,
                    sigs_reshaped_ls,
                    comps_ls,
                    comp_str,
                    fig,
                    gs[3, :],
                    legend_vloc=-1.2,
                )

                plt.subplots_adjust(
                    bottom=0.135,
                    top=0.935,
                    left=0.12,
                    right=0.93,
                    wspace=0.25,
                    hspace=1.2,
                )

                plt.savefig(
                    fpath
                    / f"vecs-combo-{comp_str}C-{fsuffix}-{window_width}w-{n_waves}n.png",
                    dpi=150,
                )
                plt.savefig(
                    fpath
                    / "pdf"
                    / f"vecs-combo-{comp_str}C-{fsuffix}-{window_width}w-{n_waves}n.pdf",
                )
                plt.close()

                fig = plt.figure(figsize=(10, 6.5))
                gs = GridSpec(
                    figure=fig,
                    nrows=4,
                    ncols=2,
                    height_ratios=[0.5, 1, 1, 1],
                    width_ratios=[1.5, 1],
                )

                plot_signal(
                    t,
                    sigs,
                    sigs_idx_ls,
                    title=title,
                    colors=TRAIN_TEST_COLORS,
                    fig=fig,
                    gs=gs[0, :],
                    legend_vloc=-14.5,
                )
                plot_2_components_1d(
                    comps_ls,
                    comp_str,
                    TRAIN_TEST_COLORS,
                    fig,
                    gs,
                    vecs=False,
                )
                plot_2_components_2d(
                    comps_ls,
                    comp_str,
                    TRAIN_TEST_COLORS,
                    fig,
                    gs[1:3, 1],
                    scatter=scatter_comps,
                )
                plot_reconstruction_error(
                    model,
                    sigs_reshaped_ls,
                    comps_ls,
                    comp_str,
                    fig,
                    gs[3, :],
                    legend_vloc=-1.2,
                )

                plt.subplots_adjust(
                    bottom=0.135,
                    top=0.935,
                    left=0.12,
                    right=0.93,
                    wspace=0.25,
                    hspace=1.2,
                )

                plt.savefig(
                    fpath
                    / f"comps-combo-{comp_str}C-{fsuffix}-{window_width}w-{n_waves}n.png",
                    dpi=150,
                )
                plt.savefig(
                    fpath
                    / "pdf"
                    / f"comps-combo-{comp_str}C-{fsuffix}-{window_width}w-{n_waves}n.pdf",
                )
                plt.close()

            print(
                f"  Two comps completed in {timedelta(seconds=time.perf_counter() - t_now)}"
            )

            print()

    ################
    # 3 COMPONENTS #
    ################

    if skip_n != 3:
        n_components = 3

        if window_width > n_components:
            t_now = time.perf_counter()
            pca_, ica_, umap_, sigs_reshaped_ls = process_signal(
                sigs,
                window_width=window_width,
                n_components=n_components,
                do_umap=do_umap,
            )

            pca, pca_comps_ls, pca_vecs_ls = pca_
            ica, ica_comps_ls, ica_vecs_ls = ica_
            umap, umap_comps_ls = umap_

            fig = plt.figure(figsize=(15, 8))

            gs = GridSpec(
                figure=fig,
                nrows=3,
                ncols=6,
                height_ratios=[1.2, 4, 4],
                width_ratios=[1, 1, 1, 0.001, 0.999, 1],
            )

            plot_signal(
                t,
                sigs,
                sigs_idx_ls,
                title=title,
                colors=TRAIN_TEST_COLORS,
                fig=fig,
                gs=gs[0, 1:-1],
                legend_vloc=-12.1
            )

            plot_components_2d(
                pca_comps_ls,
                comp_str='P',
                colors=TRAIN_TEST_COLORS,
                fig=fig,
                gs=gs,
                scatter=True,
            )

            plot_components_2d(
                ica_comps_ls,
                comp_str='I',
                colors=TRAIN_TEST_COLORS,
                fig=fig,
                gs=gs,
                scatter=True,
            )

            plot_reconstruction_error(
                pca,
                sigs_reshaped_ls,
                pca_comps_ls,
                'P',
                fig,
                gs[1, 4:],
            )

            plot_reconstruction_error(
                ica,
                sigs_reshaped_ls,
                ica_comps_ls,
                'I',
                fig,
                gs[2, 4:],
            )

            plt.subplots_adjust(
                bottom=0.15,
                top=0.925,
                left=0.05,
                right=0.95,
                wspace=0.5,
                hspace=0.65,
            )
            plt.savefig(
                fpath / f"comps-combo-2D-{fsuffix}-{window_width}w-{n_waves}n.png", dpi=150
            )
            plt.savefig(
                fpath / "pdf" / f"comps-combo-2D-{fsuffix}-{window_width}w-{n_waves}n.pdf"
            )
            plt.close()

            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(
                figure=fig,
                nrows=5,
                ncols=2,
                height_ratios=[0.65, 1, 1, 1, 1],
            )

            plot_signal(
                t,
                sigs,
                sigs_idx_ls,
                title=title,
                colors=TRAIN_TEST_COLORS,
                fig=fig,
                gs=gs[0, :],
            )

            for model, vecs_ls, comps_ls, comp_str in zip(
                [pca, ica],
                [pca_vecs_ls, ica_vecs_ls],
                [pca_comps_ls, ica_comps_ls],
                ["P", "I"],
            ):
                plot_components_1d(
                    vecs_ls,
                    comp_str,
                    TRAIN_TEST_COLORS,
                    fig,
                    gs,
                    vecs=True,
                )
                plot_reconstruction_error(
                    model,
                    sigs_reshaped_ls,
                    comps_ls,
                    comp_str,
                    fig,
                    gs[4, 0 if comp_str == "P" else 1],
                )

            plt.subplots_adjust(
                bottom=0.115,
                top=0.935,
                left=0.12,
                right=0.93,
                wspace=0.4,
                hspace=1,
            )

            plt.savefig(
                fpath / f"vecs-1D-{fsuffix}-{window_width}w-{n_waves}n.png", dpi=150
            )
            plt.savefig(
                fpath / "pdf" / f"vecs-1D-{fsuffix}-{window_width}w-{n_waves}n.pdf"
            )
            plt.close()

            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(
                figure=fig,
                nrows=5,
                ncols=2,
                height_ratios=[0.65, 1, 1, 1, 1],
            )

            plot_signal(
                t,
                sigs,
                sigs_idx_ls,
                title=title,
                colors=TRAIN_TEST_COLORS,
                fig=fig,
                gs=gs[0, :],
            )

            for model, vecs_ls, comps_ls, comp_str in zip(
                [pca, ica],
                [pca_vecs_ls, ica_vecs_ls],
                [pca_comps_ls, ica_comps_ls],
                ["P", "I"],
            ):
                plot_components_1d(
                    comps_ls,
                    comp_str,
                    TRAIN_TEST_COLORS,
                    fig,
                    gs,
                    vecs=False,
                )
                plot_reconstruction_error(
                    model,
                    sigs_reshaped_ls,
                    comps_ls,
                    comp_str,
                    fig,
                    gs[4, 0 if comp_str == "P" else 1],
                )

            plt.subplots_adjust(
                bottom=0.115,
                top=0.935,
                left=0.12,
                right=0.93,
                wspace=0.4,
                hspace=1,
            )

            plt.savefig(
                fpath / f"comps-1D-{fsuffix}-{window_width}w-{n_waves}n.png", dpi=150
            )
            plt.savefig(
                fpath / "pdf" / f"comps-1D-{fsuffix}-{window_width}w-{n_waves}n.pdf"
            )
            plt.close()

            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(
                figure=fig,
                nrows=3,
                ncols=3,
                height_ratios=[0.25, 1, 1],
            )

            plot_signal(
                t,
                sigs,
                sigs_idx_ls,
                title=title,
                colors=TRAIN_TEST_COLORS,
                fig=fig,
                gs=gs[0, :],
                legend_vloc=-13.85,
            )

            for comps_ls, comp_str in zip([pca_comps_ls, ica_comps_ls], ["P", "I"]):
                plot_components_2d(
                    comps_ls, comp_str, TRAIN_TEST_COLORS, fig, gs, scatter=scatter_comps
                )

            plt.subplots_adjust(
                bottom=0.125,
                top=0.935,
                left=0.11,
                right=0.94,
                hspace=0.65,
                wspace=0.5,
            )

            plt.savefig(
                fpath / f"comps-2D-{fsuffix}-{window_width}w-{n_waves}n.png", dpi=150
            )
            plt.savefig(
                fpath / "pdf" / f"comps-2D-{fsuffix}-{window_width}w-{n_waves}n.pdf"
            )
            plt.close()

            fig = plt.figure(figsize=(10, 8))
            gs = GridSpec(
                figure=fig,
                nrows=3,
                ncols=3,
                height_ratios=[0.25, 1, 1],
            )

            plot_signal(
                t,
                sigs,
                sigs_idx_ls,
                title=title,
                colors=TRAIN_TEST_COLORS,
                fig=fig,
                gs=gs[0, :],
                legend_vloc=-13.85,
            )

            for vecs_ls, comp_str in zip([pca_vecs_ls, ica_vecs_ls], ["P", "I"]):
                plot_components_2d(
                    vecs_ls,
                    comp_str,
                    TRAIN_TEST_COLORS,
                    fig,
                    gs,
                    scatter=scatter_comps,
                    vecs=True,
                )

            plt.subplots_adjust(
                bottom=0.125,
                top=0.935,
                left=0.11,
                right=0.94,
                hspace=0.65,
                wspace=0.5,
            )

            plt.savefig(
                fpath / f"vecs-2D-{fsuffix}-{window_width}w-{n_waves}n.png", dpi=150
            )
            plt.savefig(
                fpath / "pdf" / f"vecs-2D-{fsuffix}-{window_width}w-{n_waves}n.pdf"
            )
            plt.close()

            if umap:
                fig = plt.figure(figsize=(10, 5))
                gs = GridSpec(
                    ncols=3,
                    width_ratios=[1, 1, 1],
                    nrows=2,
                    height_ratios=[0.3, 1],
                )

                plot_signal(
                    t,
                    sigs,
                    sigs_idx_ls,
                    title=title,
                    colors=TRAIN_TEST_COLORS,
                    fig=fig,
                    gs=gs[0, :],
                    legend_vloc=-6.4,
                )

                plot_components_2d(
                    umap_comps_ls,
                    "UMAP",
                    TRAIN_TEST_COLORS,
                    fig,
                    gs,
                    scatter=True,
                )

                plt.subplots_adjust(
                    top=0.915,
                    bottom=0.2,
                    left=0.11,
                    right=0.94,
                    hspace=0.725,
                    wspace=0.45,
                )

                plt.savefig(
                    fpath / f"comps-2D-umap-{fsuffix}-{window_width}w-{n_waves}n.png",
                    dpi=150,
                )
                plt.savefig(
                    fpath
                    / "pdf"
                    / f"comps-2D-umap-{fsuffix}-{window_width}w-{n_waves}n.pdf",
                    dpi=150,
                )
                plt.close()

            print(
                f"  Three comps completed in {timedelta(seconds=time.perf_counter() - t_now)}"
            )

            print()


def main():
    ...


if __name__ == "__main__":
    main()
