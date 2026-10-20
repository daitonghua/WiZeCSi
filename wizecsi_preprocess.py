"""WiZeCSi: multi-ratio augmentation for Widar3 raw CSI.

Every raw Widar3 measurement is a ``(T, 90)`` complex matrix, i.e. 3 receive
antennas x 30 subcarriers.  For every ordered pair of receive antennas this
script forms one CSI ratio (Eq. 6) and turns it into one
Single-Link-Single-Ratio (SLSR) sample (Eq. 10): the amplitude and the phase of
the ratio concatenated along the frequency axis.

With ``N = 3`` antennas there are ``M = N(N - 1) = 6`` ratios, so each raw file
yields six SLSR samples of shape ``(T_MAX, 2 * F) = (3000, 60)``.
"""

import argparse
import os

import numpy as np
import scipy.io as sio

# N(N - 1) ordered antenna pairs -> the six CSI ratios of the paper (Eq. 9)
ANT_PAIRS = [[1, 2], [2, 3], [3, 1], [1, 3], [2, 1], [3, 2]]
N_ANT = 3
N_SC_PER_ANT = 30
N_RATIO = len(ANT_PAIRS)

T_MAX = 3000            # packets kept per sample
N_SC = N_SC_PER_ANT     # subcarriers per SLSR sample
AMP, PHA = N_SC, N_SC   # column layout of an SLSR sample: |H| then angle(H)

# subcarrier indices of the Intel 5300 (used to remove the linear phase offset)
SC_INDEX = np.array([-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4,
                     -2, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 28])
SC_SPAN = float(SC_INDEX[-1] - SC_INDEX[0])

CLASSES = [1, 2, 3, 4, 6, 9]
USERS = [str(i) for i in range(11, 17)]
LOCS = [str(i) for i in range(1, 6)]
ORIS = [str(i) for i in range(1, 6)]
REPS = [str(i) for i in range(1, 6)]

DEF_LOAD_ROOT = os.path.join("data", "widar3")
DEF_SAVE_ROOT = os.path.join("data", "widar3_slsr")


def minmax_to_pm1(x):
    """Rescale ``x`` to [-1, 1] (a flat input maps to all zeros)."""
    lo, hi = np.min(x), np.max(x)
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo) * 2.0 - 1.0


def phase_compensation(ang):
    """Remove the random linear phase offset caused by STO / SFO / CFO.

    The ratio of two antennas on the same NIC still carries a random phase
    slope over the frequency axis.  Unwrapping along the subcarriers and
    subtracting the line fitted between the two edge subcarriers plus the mean
    level of the measurement cancels it, leaving the phase rotation produced by
    the moving hand.
    """
    out = np.unwrap(ang, axis=1)
    slope = (out[:, -1] - out[:, 0]) / SC_SPAN
    out = out - SC_INDEX[None, :] * slope[:, None] - out.mean()
    return (out + np.pi) % (2.0 * np.pi) - np.pi


def zero_padding(x, t_max=T_MAX, n_col=N_SC, n_fill=0.0):
    """Pad or truncate ``x`` along time to ``(t_max, n_col)``."""
    out = np.full((t_max, n_col), n_fill, dtype=np.float32)
    t = min(x.shape[0], t_max)
    out[:t] = x[:t]
    return out


def antenna_blocks(csi):
    """Split a ``(T, 90)`` raw CSI matrix into the three per-antenna blocks."""
    return [csi[:, a * N_SC_PER_ANT:(a + 1) * N_SC_PER_ANT] for a in range(N_ANT)]


def slsr_sample(ratio):
    """Turn one complex CSI ratio into its ``(T_MAX, 60)`` SLSR sample."""
    amp = minmax_to_pm1(np.abs(ratio))
    ang = minmax_to_pm1(phase_compensation(np.angle(ratio)))
    return np.concatenate((zero_padding(amp, n_col=AMP),
                           zero_padding(ang, n_col=PHA)), axis=1)


def load_csi(mat_path):
    """Load ``csi_data`` and repair the entries whose magnitude vanished."""
    csi = sio.loadmat(mat_path)["csi_data"]
    if csi.size == 0 or not np.any(csi):                     # lost measurement
        csi = (np.random.randn(T_MAX, N_ANT * N_SC_PER_ANT)
               + 1j * np.random.randn(T_MAX, N_ANT * N_SC_PER_ANT))
    csi = np.asarray(csi, dtype=np.complex128)
    csi[np.abs(csi) == 0] = np.random.randn() + 1j * np.random.randn()
    return csi


def process_one(mat_path, out_dir, stem):
    """Write the ``N_RATIO`` SLSR samples of one raw measurement."""
    blocks = antenna_blocks(load_csi(mat_path))
    for i, (ant_1, ant_2) in enumerate(ANT_PAIRS):
        sample = slsr_sample(blocks[ant_1 - 1] / blocks[ant_2 - 1])
        np.save(os.path.join(out_dir, f"{stem}-{i + 1}.npy"), sample)


def file_sel(names, users, locs, oris, reps):
    """Keep the raw file names matching the requested domain factors."""
    out = []
    for name in names:
        if not name.endswith(".mat"):
            continue
        parts = name[:-4].split("-")            # user<u>-<label>-<loc>-<ori>-<rep>-r<rx>
        if len(parts) < 6:
            continue
        if (parts[0][4:] in users and parts[2] in locs
                and parts[3] in oris and parts[4] in reps):
            out.append(name)
    return sorted(out)


def run(cfg):
    n_file = 0
    for cl in cfg.classes:
        src_dir = os.path.join(cfg.load_root, str(cl))
        if not os.path.isdir(src_dir):
            print(f"[skip] missing class directory: {src_dir}")
            continue
        dst_dir = os.path.join(cfg.save_root, str(cl))
        os.makedirs(dst_dir, exist_ok=True)

        names = file_sel(os.listdir(src_dir), cfg.users, cfg.locs, cfg.oris, cfg.reps)
        if cfg.limit:
            names = names[:cfg.limit]
        for name in names:
            process_one(os.path.join(src_dir, name), dst_dir, name[:-4])
            n_file += 1
        print(f"class {cl}: {len(names)} raw files -> {len(names) * N_RATIO} SLSR samples")
    print(f"\ndone: {n_file} raw files -> {n_file * N_RATIO} samples in {cfg.save_root}")


def build_parser():
    p = argparse.ArgumentParser(description="WiZeCSi: multi-ratio augmentation on Widar3")
    p.add_argument("--load-root", default=DEF_LOAD_ROOT, help="root of the raw .mat files")
    p.add_argument("--save-root", default=DEF_SAVE_ROOT, help="output root of the SLSR samples")
    p.add_argument("--classes", type=int, nargs="+", default=CLASSES)
    p.add_argument("--users", nargs="+", default=USERS)
    p.add_argument("--locs", nargs="+", default=LOCS)
    p.add_argument("--oris", nargs="+", default=ORIS)
    p.add_argument("--reps", nargs="+", default=REPS)
    p.add_argument("--limit", type=int, default=0,
                   help="process only the first K files of each class (0 = all)")
    p.add_argument("--seed", type=int, default=None, help="optional random seed")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
