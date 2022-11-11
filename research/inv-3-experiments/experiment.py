from gmutils_v2 import *
from scipy.signal import stft, istft
import streamlit as st

fpath = Path(__file__)
root = fpath.parents[0]


def low_freq(n_waves=10):
    print("\nLow frequency data\n")
    t_start = time.perf_counter()

    fpathi = fpath.parents[0] / "Data" / "lf"

    f_max = 50

    f = 25

    fs_down = 1500

    for ff in fpathi.glob("*.csv"):
        df = pd.read_csv(ff)

        df["acc_m"] = np.nanmean([df.acc.shift(-1), df.acc.shift(1)], axis=0)
        df.acc[df.acc.isna()] = df.acc_m

        t = df.t.values - df.t.iloc[0]
        sig = df.acc.values
        fs = 1 / df.dt.mean()

        step = int(fs / fs_down)
        if step < 1:
            fs_down = int(fs)
            step = 1

        print(
            f"\n {ff.stem}, {n_waves = }, {fs_down = }"
        )

        ti = t[::step]
        sigi = sig[::step]

        fsuffix = f"{ff.stem.replace('.', '_')}-{fs_down}Hz"
        title = f"Experimental Data: {ff.stem}"

        w = 2 * np.pi * f
        generate_results(
            ti,
            sigi,
            n_waves,
            w,
            title=title,
            fpath=root / "LowFreq",
            fsuffix=fsuffix,
            train_test_ratios=(1, 1, 1),
            skip_n=2,
        )

    print(
        f"Low frequency data completed in {timedelta(seconds=time.perf_counter() - t_start)}"
    )


def high_freq(n_waves=10):
    print("\nHigh frequency data\n")
    t_start = time.perf_counter()

    fpathi = fpath.parents[0] / "Data" / "hf"

    fs_down = 1500

    f = 25
    f_max = 50

    for ff in fpathi.glob("*.csv"):
        df = pd.read_csv(ff)
        fs = 1 / df.dt.mean()
        t = df.t.values - df.t.iloc[0]
        sig = df.acc.values

        step = int(fs / fs_down)
        if step < 1:
            fs_down = int(fs)
            step = 1
        print(f"\n {ff.stem}, {n_waves = }, {fs_down = }")

        ti = t[::step]
        sigi = sig[::step]

        fsuffix = f"{ff.stem.replace('.', '_')}-{fs_down}Hz"
        title = f"Experimental Data: {ff.stem}"

        w = 2 * np.pi * f
        print("  Train-test split: (1, 1, 1)")
        try:
            generate_results(
                ti,
                sigi,
                n_waves,
                w,
                title=title,
                fpath=root / "HighFreq",
                fsuffix=fsuffix,
                train_test_ratios=(1, 1, 1),
                skip_n=2,
            )
        except MemoryError:
            print("MemoryError. Skipping.\n")

    print(
        f"High frequency data completed in {timedelta(seconds=time.perf_counter() - t_start)}"
    )


def main():
    t_start = time.perf_counter()

    print(f"\n{'=' * 50}\n")

    low_freq(n_waves=30)

    print(f"\n{'=' * 50}\n")

    high_freq(n_waves=20)

    print(f"\n{'=' * 50}\n")

    print(f"\nTotal time to run: {timedelta(seconds=time.perf_counter() - t_start)}")


if __name__ == "__main__":
    root = fpath.parents[0]
    root /= "Plots"
    print(" / ".join(root.parts[-3:]))

    main()
