from gmutils_v2 import *

E = "e"
F = "f"
D = "d"
P = "%"

A = 1
F = 50
P = 0
M = 0

fpath = Path(__file__)

sym_dict = {"a": "a", "f": "f", "w": "\omega", "p": "\phi", "m": "m"}
units_dict = {"a": "", "w": "rad/s", "f": " Hz", "p": " rad", "m": ""}
pi_str = f"$\pi$"

base_sig = box({"a": A, "f": F, "p": P, "m": M, "noise": True})


def merge_dicts(d1, d2):
    d3 = d1.copy()
    for k, v in d2.items():
        if k not in d1.keys():
            d3[k] = v
    return d3


def sort_params(d):
    p_ls = ["a", "f", "p", "m"]
    return [(p, d[p]) for p in p_ls if p in d.keys()]


def format_value(p, v):
    return f"{sf_str(v * 2 if p == 'p' else v, 3)}{pi_str if p == 'p' else ''}"


def sig_str(new_sig, base_sig=base_sig, mpl=True):
    def get_str(p, new_sig, base_sig):
        param_str = (
            sf_str(new_sig[p], 3) if p == "f" or new_sig[p] != base_sig[p] else ""
        )
        return param_str.rstrip("0").rstrip(".") if "." in param_str else param_str

    new_sig = merge_dicts(new_sig, base_sig)

    a_str = get_str("a", new_sig, base_sig)
    f_str = get_str("f", new_sig, base_sig)
    m_str = get_str("m", new_sig, base_sig)
    p_str = get_str("p", new_sig, base_sig)

    times = "\\times"

    if mpl:
        f_str = f"{f_str}{times} " if f_str != "" else ""
        m_str = f" + {m_str}" if m_str != "" else ""
        p_str = (
            f" + {f'{p_str}{times}' if p_str != '1' else ''}2\\pi"
            if p_str != ""
            else ""
        )
        return f"${a_str}\ \sin({f_str}2\\pi\ t{p_str}){m_str}$"

    a_str = f"{a_str} " if a_str != "" else ""
    f_str = f"{f_str} * " if f_str != "" else ""
    m_str = f" + {m_str}" if m_str != "" else ""
    p_str = f" + {f'{p_str} * ' if p_str != '1' else ''}2 pi" if p_str != "" else ""
    return f"{a_str}sin({f_str}2 pi t{p_str}){m_str}"


def generate_signal(t, *, a=A, f=F, p=P, m=M, noise=False):
    sig = a * np.sin(2 * np.pi * f * t + 2 * np.pi * p) + m
    noise_sig = a * np.random.uniform(-0.1, 0.1, t.shape)
    return sig + noise_sig if noise else sig


def combine_signals(sig1, sig2, stepped_frac):
    n_points = sig1.size
    sigc = sig1.copy()

    sig1_start = int(n_points * (1 - stepped_frac))
    sig2_stop = int(n_points * stepped_frac)

    sigc[sig1_start:] = sig2[:sig2_stop]

    return sigc


def step_signal(step_params, n_points, start_params=None, stepped_frac=0.5, ti=0, tf=1):
    if not start_params:
        start_params = base_sig

    ts = tf / n_points

    t1 = np.arange(ti, tf, ts)
    sig1 = generate_signal(t1, **start_params)

    t2 = np.arange(ti, tf + ts, ts)
    sig2 = generate_signal(t2, **step_params)

    tc = t1
    sigc = combine_signals(sig1, sig2, stepped_frac)

    return (tc, sigc), (t1, sig1), (t2, sig2)


def sweep_signal(sig_params, n_points, ti=0, tf=1):
    ts = tf / n_points

    t = np.arange(ti, tf, ts)
    sig = generate_signal(t, **sig_params)

    return t, sig


def constants_bulk(const_param_ls, n_points, n_waves=10, ti=0, tf=1):
    print("\nConstant values\n")
    t_now = time.perf_counter()

    for params_ in const_param_ls:
        params = merge_dicts(params_, base_sig)
        t = np.linspace(ti, tf, n_points)
        sig = generate_signal(t, **params)
        params.pop("noise")
        if "noise" in params_.keys():
            params_.pop("noise")
        print(sig_str(params, mpl=False), end="\n\n")
        title_str = sig_str(params)
        fsuffix = "-".join(
            [
                "const",
                *[
                    f"{p}{sf_str(v, 3)}".replace(".", "_")
                    for p, v in sort_params(params_)
                ],
            ]
        )

        params_str = "-".join(p for p, _ in sort_params(params_))

        w = 2 * np.pi * params.f
        generate_results(
            t,
            sig,
            n_waves,
            w,
            title=f"Signal: {title_str}",
            fpath=root / "constant" / params_str,
            fsuffix=fsuffix,
            train_test_ratios=(1,),
            do_umap=True,
            skip_n=3,
        )

    print(
        f"Constant values completed in: {timedelta(seconds=time.perf_counter() - t_now)}"
    )


def step_bulk(step_param_ls, n_points, n_waves, stepped_frac=0.5):
    print("\nStepping\n")
    t_now = time.perf_counter()

    for step_params_ in step_param_ls:
        step_params = merge_dicts(step_params_, base_sig)
        (tc, sigc), _, _ = step_signal(
            step_params, n_points, stepped_frac=stepped_frac
        )
        step_params.pop("noise")
        if "noise" in step_params_.keys():
            step_params_.pop("noise")
        print(sig_str(step_params, mpl=False), end="\n\n")
        title_str = sig_str(step_params)
        fsuffix = "-".join(
            [
                "step",
                *[
                    f"{p}{sf_str(v, 3)}".replace(".", "_")
                    for p, v in sort_params(step_params_)
                ],
            ],
        )

        params_str = "-".join(p for p, _ in sort_params(step_params_))

        w = 2 * np.pi * base_sig.f
        print(" Train-test split: (1, 1, 1)")
        generate_results(
            tc,
            sigc,
            n_waves,
            w,
            title=f"Stepped signal: {title_str}"
            if title_str != sig_str(base_sig)
            else "Base signal",
            fpath=root / "step" / params_str,
            fsuffix=fsuffix,
            train_test_ratios=(1, 1, 1),
            do_umap=True,
        )

    print(f"Stepping completed in: {timedelta(seconds=time.perf_counter() - t_now)}")


def sweep_bulk(sweep_param_ls, n_points, n_waves, step=False, decr=False):
    print("\nSweeping\n")
    t_now = time.perf_counter()
    if decr and step:
        return

    for sweep_params_ in sweep_param_ls:
        sweep_params = merge_dicts(sweep_params_, base_sig)
        t, sig = sweep_signal(sweep_params, n_points)
        sig = sig[::-1] if decr else sig
        sweep_params.pop("noise")
        print(box(sort_params(sweep_params_)), end="\n\n")
        sweep_params_ranges_ls = [
            f"${sym_dict[p]}$ $\in$ "
            f"[{format_value(p, min(abs(v)))}, {format_value(p, max(abs(v)))}]{units_dict[p]}"
            for p, v in sort_params(sweep_params_)
        ]
        params_str = "; ".join(sweep_params_ranges_ls)
        fsuffix = (
            "-".join(
                [
                    "sweep-step-decr",
                    *[p for p, _ in sort_params(sweep_params_)],
                ]
            )
            if step and decr
            else "-".join(
                [
                    "sweep-step",
                    *[p for p, _ in sort_params(sweep_params_)],
                ]
            )
            if step
            else "-".join(
                [
                    "sweep-decr",
                    *[p for p, _ in sort_params(sweep_params_)],
                ]
            )
            if decr
            else "-".join(
                [
                    "sweep",
                    *[p for p, _ in sort_params(sweep_params_)],
                ]
            )
        )

        params_folder_str = "-".join(p for p, _ in sort_params(sweep_params_))

        f = sweep_params.f
        f = (
            f
            if isinstance(f, (float, int))
            else (sweep_params.f[-1] if decr else sweep_params.f[0])
        )
        w = 2 * np.pi * f
        print(" Train-test split: (1, 1, 1)")
        generate_results(
            t,
            sig,
            n_waves,
            w,
            title=f"Sweep parameters: {params_str}",
            fpath=root / "sweep" / params_folder_str,
            fsuffix=fsuffix,
            train_test_ratios=(1, 1, 1),
            do_umap=True,
            skip_n=2,
        )

    print(f"Sweeping completed in: {timedelta(seconds=time.perf_counter() - t_now)}")


def step_sweep_decr_bulk(sweep_param_ls, n_points, n_waves):
    print("\nSweeping\n")
    t_now = time.perf_counter()

    for start_params_, sweep_params_ in sweep_param_ls:
        sweep_params = merge_dicts(sweep_params_, base_sig)
        start_params = merge_dicts(start_params_, base_sig)
        print(sweep_params)
        t, sig = sweep_signal(sweep_params, n_points)
        sig = np.hstack([generate_signal(t, **start_params)[::2], -sig[::-2]])
        sweep_params_c = sweep_params_.copy()
        for k, v in sweep_params_c.items():
            if v.min() == v.max():
                sweep_params_.pop(k)
        sweep_params.pop("noise")
        start_params.pop("noise")
        sweep_params.f *= 2
        start_params.f *= 2
        print(box(sort_params(sweep_params_)), end="\n\n")
        sweep_params_ranges_ls = [
            f"${sym_dict[p]}$ $\in$ "
            f"[{format_value(p, min(abs(v)))}, {format_value(p, max(abs(v)))}]{units_dict[p]}"
            for p, v in sort_params(sweep_params_)
        ]
        params_str = "; ".join(sweep_params_ranges_ls)
        fsuffix = "-".join(
            [
                "sweep-step-decr",
                *[p for p, _ in sort_params(sweep_params_)],
            ]
        )

        params_folder_str = "-".join(p for p, _ in sort_params(sweep_params_))

        f = start_params.f
        f = f if isinstance(f, (float, int)) else start_params.f[0]
        w = 2 * np.pi * f * 2
        print(" Train-test split: (1, 1, 1)")
        generate_results(
            t,
            sig,
            n_waves,
            w,
            title=f"Sweep parameters: {params_str}",
            fpath=root / "sweep" / params_folder_str,
            fsuffix=fsuffix,
            train_test_ratios=(1, 1, 1),
            do_umap=True,
            skip_n=2,
        )

    print(f"Sweeping completed in: {timedelta(seconds=time.perf_counter() - t_now)}")


def main():
    t_start = time.perf_counter()

    n_points = 3000
    n_waves = 5

    samples = partial(np.linspace, num=n_points)
    samples_half = partial(np.linspace, num=(n_points // 2))

    print(f"\n{'=' * 50}\n")

    const_param_ls = [
        *[box({"a": v}) for v in A * np.array([0.5, 1, 5])],
        *[box({"f": v}) for v in F * np.array([0.5, 1, 2.5])],
        *[box({"p": v}) for v in P + np.linspace(0, 1, 9)],
        *[box({"m": v}) for v in M + np.array([0, 1, 5])],
    ]
    constants_bulk(const_param_ls, n_points)

    print(f"\n{'=' * 50}\n")

    step_param_ls = [
        *[box({"a": v}) for v in A * np.array([0.25, 0.5, 1, 2.5, 5])],
        *[box({"f": v}) for v in F * 3 / 5 * np.array([0.25, 0.5, 1, 2.5])],
        *[
            box({"f": vf, "a": va}) for vf, va in zip(
                F * 3 / 5 * np.array([0.25, 0.5, 0.75, 1, 1.5, 2, 2.5]),
                A * np.array([0.25, 0.5, 1, 2.5, 5])
            )
        ],
        *[box({"p": v}) for v in P + np.linspace(0, 1, 9)],
        *[box({"m": v}) for v in M + np.array([0, 1, 2.5, 5, 10])],
        *[
            box({"f": vf, "a": va})
            for vf, va in zip(
                np.array([7.5, 7.5, 75, 75]), np.array([0.25, 5, 0.25, 5])
            )
        ],
    ]
    step_bulk(step_param_ls, n_points, n_waves, stepped_frac=0.5)

    print(f"\n{'=' * 50}\n")

    sweep_param_ls = [
        box({"a": samples(0, 10)}),
        box({"f": samples(25, 50)}),
        box({"f": samples(25, 50)}),
        box({"p": samples(0, 1)}),
        box({"m": samples(-5, 5)}),
        box({"a": samples(0, 10), "f": samples(10, 50)}),
    ]
    sweep_bulk(sweep_param_ls, n_points, n_waves)

    print(f"\n{'=' * 50}\n")

    sweep_bulk(sweep_param_ls, n_points, n_waves, decr=True)

    print(f"\n{'=' * 50}\n")

    sweep_param_ls = [
        box({"a": np.hstack([samples_half(A, A), samples_half(A, 5)])}),
        box(
            {
                "f": np.hstack(
                    [
                        samples_half(F // 2, F // 2),
                        samples_half(F // 2, 50),
                    ]
                )
            }
        ),
        box({"p": np.hstack([samples_half(P, P), samples_half(P, 1)])}),
        box({"m": np.hstack([samples_half(M, M), samples_half(M, 5)])}),
        box(
            {
                "f": np.hstack(
                    [
                        samples_half(F // 2, F // 2),
                        samples_half(F // 2, 50),
                    ]
                ),
                "a": np.hstack([samples_half(A, A), samples_half(A, 5)]),
            }
        ),
    ]
    sweep_bulk(sweep_param_ls, n_points, n_waves, step=True)

    print(f"\n{'=' * 50}\n")

    sweep_param_ls = [
        [
            box({"f": samples(F, F)}),
            box({"f": samples(0, F // 2), "a": samples(A, A)}),
        ],
        [
            box({"f": samples(F, F), 'a': samples(5, 5)}),
            box({"f": samples(F, F), "a": samples(0, 5)}),
        ],
        [
            box({"f": samples(F, F), "a": samples(5, 5)}),
            box({
                "f": samples(0, F // 2),
                "a": samples(0, 5),
            }),
        ],
    ]
    step_sweep_decr_bulk(sweep_param_ls, n_points, n_waves)

    print(f"\n{'=' * 50}\n")

    print(f"\nTotal time to run: {timedelta(seconds=time.perf_counter() - t_start)}")


if __name__ == "__main__":
    root = fpath.parents[0]
    root /= "Plots"
    print(" / ".join(root.parts[-3:]))

    main()
