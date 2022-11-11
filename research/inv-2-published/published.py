from gmutils_v2 import *

fpath = Path(__file__)
root = fpath.parents[0]

data_path = Path("./data/")


def huang_baddour(n_waves=10):
    print("\nHuang & Baddour Dataset\n")
    t_start = time.perf_counter()

    files = [
        ["H-A-1", (14.1, 23.8)],
        ["O-A-1", (14.8, 27.1)],
    ]

    fs = 200_000
    t_max = 10

    fs_down = 1_500
    step = fs // fs_down

    for ds_str, (f_min, _) in files:
        t, sig, _ = pd.read_csv(fpath.parents[0] / data_path / f"{ds_str}.csv").values.T
        t_start, t_end = 0, 10
        print(f"\n {ds_str}: {fs_down}Hz, {t_start}-{t_end}s")

        idx = np.arange(
            start=sig.size * t_start // t_max,
            stop=sig.size * t_end // t_max,
            step=step,
        )
        ti = t[idx]
        sigi = sig[idx]

        f = f_min

        fsuffix = f"{ds_str}-{t_start}-{t_end}-{fs_down}Hz"
        title = f"Huang & Baddour (2018): {ds_str} ({t_start}-{t_end} s)"

        w = 2 * np.pi * f
        print("  Train-test split: (1, 1, 1)")
        try:
            generate_results(
                ti,
                sigi,
                n_waves,
                w,
                title=title,
                fpath=root / "HuangBaddour" / ds_str,
                fsuffix=fsuffix,
                train_test_ratios=(1, 1, 1),
                skip_n=2,
            )
        except MemoryError:
            print("MemoryError. Skipping.\n")

    df_healthy = pd.read_csv(fpath.parents[0] / data_path / f"{files[0][0]}.csv")
    t = np.linspace(0, 20, 20 * fs)
    for ds_str_, (f_min, _) in files[1:]:
        df_unhealthy = pd.read_csv(fpath.parents[0] / data_path / f"{ds_str_}.csv")
        df = pd.concat([df_healthy, df_unhealthy])
        _, sig, _ = df.values.T

        ds_str = f"H-A-1-vs-{ds_str_}"
        print(f"\n {ds_str}: {fs_down = }Hz\n")

        ti = t[::step]
        sigi = sig[::step]

        f = f_min

        fsuffix = f"{ds_str}-{fs_down}Hz"
        title = f"Huang & Baddour (2018): H-A-1 vs. {ds_str_}"

        w = 2 * np.pi * f
        print("  Train-test split: (1, 1)")
        try:
            generate_results(
                ti,
                sigi,
                n_waves,
                w,
                title=title,
                fpath=root / "HuangBaddour" / ds_str,
                fsuffix=fsuffix,
                train_test_ratios=(1, 1),
                skip_n=2,
            )
        except MemoryError:
            print("MemoryError. Skipping.\n")
            return

    print(
        f"Huang & Baddour completed in {timedelta(seconds=time.perf_counter() - t_start)}"
    )


def ims(n_waves=10):
    print("\nIMS Dataset\n")
    t_start = time.perf_counter()

    f = 2000 / 60
    w = 2 * np.pi * f

    fs = 20_480
    t = np.linspace(0, 1, fs)

    fs_down = 1_536
    step = fs // fs_down

    for ff in (fpath.parents[0] / data_path).glob("*ims*.csv"):
        df = pd.read_csv(ff).T
        ds_str = ff.stem.split("-")[1]
        for col, sig in df.iterrows():
            if col not in ["b1"]:
                continue

            print(f"\n {ds_str} {col}: {fs_down}Hz, {n_waves = }")
            ti = t[::step]
            sigi = sig.values[::step]

            fsuffix = f"{ds_str}-{col}-{fs_down}Hz"
            title = f"Qiu, Lee, et al. - IMS Dataset (2006): {ds_str.capitalize()}"
            print("  Train-test split: (1, 1, 1)")
            try:
                generate_results(
                    ti,
                    sigi,
                    n_waves,
                    w,
                    title=title,
                    fpath=root / "IMS" / ds_str,
                    fsuffix=fsuffix,
                    train_test_ratios=(1, 1, 1),
                    skip_n=2,
                )
            except MemoryError:
                print("MemoryError. Skipping.\n")

    print("Healthy vs Unhealthy")

    df_healthy = pd.read_csv(
        fpath.parents[0] / data_path / "ims-healthy-2004.02.12.10.32.39.csv"
    )
    df_unhealthy = pd.read_csv(
        fpath.parents[0] / data_path / "ims-unhealthy-2004.02.19.06.02.39.csv"
    )
    df = pd.concat([df_healthy, df_unhealthy]).T
    t = np.linspace(0, 2, 2 * fs)

    ds_str = "healthy-train-unhealthy-test"

    for col, sig in df.iterrows():
        if col not in ["b1"]:
            continue
        print(f"\n {ds_str} {col}: {fs_down = }Hz, {n_waves = }\n")
        ti = t[::step]
        sigi = sig.values[::step]

        fsuffix = f"{ds_str}-{col}-{fs_down}Hz"
        title = f"Qiu, Lee, et al. - IMS Dataset (2006)"

        print("  Train-test split: (1, 1)")
        try:
            generate_results(
                ti,
                sigi,
                n_waves,
                w,
                title=title,
                fpath=root / "IMS" / ds_str,
                fsuffix=fsuffix,
                train_test_ratios=(1, 1),
                skip_n=2,
            )
        except MemoryError:
            print("MemoryError. Skipping.\n")

    print(f"IMS completed in {timedelta(seconds=time.perf_counter() - t_start)}")


def main():
    t_start = time.perf_counter()

    print(f"\n{'=' * 50}\n")

    ims(n_waves=20)

    print(f"\n{'=' * 50}\n")

    huang_baddour(n_waves=25)

    print(f"\n{'=' * 50}\n")

    print(f"\nTotal time to run: {timedelta(seconds=time.perf_counter() - t_start)}")


if __name__ == "__main__":
    root = fpath.parents[0]
    root /= "Plots"
    print(" / ".join(root.parts[-3:]))

    main()
