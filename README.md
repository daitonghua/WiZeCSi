# WiZeCSi

**WiZeCSi** — a Wi-Fi-based single-link zero-effort cross-domain gesture
recognition system. Reference implementation of

> T. Dai and E. Khorov, "WiZeCSi: Toward Wi-Fi-Based Single-Link Zero-Effort
> Cross-Domain Gesture Recognition," *IEEE Access*, vol. 14, 2026.
> DOI: [10.1109/ACCESS.2026.3677814](https://doi.org/10.1109/ACCESS.2026.3677814)

The key idea is that a **Single-Link-Single-Ratio (SLSR)** sample is already a
strong input, so the system maximizes the accuracy of one link instead of fusing
links inside the network. Two data methods make that work:

* **Multi-ratio augmentation** — an `N`-antenna receiver produces `M = N(N-1)`
  CSI ratios (Eq. 9). Since all of them are different Möbius transforms of the
  same dynamic path, every one is a valid sample. With `N = 3` that is 6 samples
  per measurement.
* **Link mixing** — SLSR samples from all receivers go into a single dataset with
  no link label, so one model serves any link, any link combination, and even a
  link it never saw in training.

`M × L` samples (6 ratios × 6 links = 36) belong to one gesture, and their
logits are averaged at test time (late fusion, Eq. 11). A modified AlexNet with
a LayerNorm before the output layer classifies each sample (§IV-D): **20.6 M
parameters, 1.21 GFLOPs per SLSR sample** — reproduce with
`python wizecsi_models.py`. `--n-sz 1` gives a half-width variant (9.9 M
parameters, 0.34 GFLOPs) for quick experiments; the depthwise-separable model of
§V-F costs 0.24 GFLOPs at the same accuracy.

## Quick start with the Widar3 dataset

Given the Widar3 raw files at `<raw>/<class>/user<u>-<label>-<loc>-<ori>-<rep>-r<rx>.mat`
(key `csi_data`, shape `(T, 90)` = 3 antennas × 30 subcarriers), build the SLSR
dataset and then train:

```bash
pip install -r requirements.txt

python wizecsi_preprocess.py --load-root <raw> --save-root data/widar3_slsr
python wizecsi_train.py     --data-root data/widar3_slsr
```

`wizecsi_preprocess.py` writes 6 SLSR samples of shape `(3000, 60)` per raw
file: 30 amplitude + 30 phase subcarriers of one CSI ratio, zero-padded to
`T_MAX = 3000`. Use `--limit 100` on it for a quick dry run, and `--help` on
either script for the rest of the options.

Training uses all six links and all six ratios at once (link mixing +
multi-ratio augmentation), then validates on the held-out repetition with a
**single link** at a time — the setting the paper targets. Only one link is used
for each prediction, and its six ratios are fused per gesture instance.

A smaller run that finishes quickly:

```bash
python wizecsi_train.py --data-root data/widar3_slsr \
    --users 11 12 --locs 1 --oris 1 --links 1 2 --ratios 1 2 \
    --train-reps 1 2 3 4 --test-reps 5 --epochs 8 --bs 240
```

Add `--test-users <u> ...` with users that were preprocessed but not trained on
to get the cross-user/cross-room number.

## Reference results on Widar3 (single link)

Six gestures, six users, 5 locations × 5 orientations, one repetition held out;
accuracy is the late-fused instance accuracy.

| setting | links | accuracy |
| --- | --- | --- |
| in-domain | 1 | 98.32% |
| in-domain | 6 | 99.09% |
| cross-location | 1 | > 97% |
| cross-orientation | 1 | 96.57% |
| cross-room-user | 1 | 96.86% |
| cross-location-orientation-room-user | 1 | 91.56% |

## Files

| file | contents |
| --- | --- |
| `wizecsi_preprocess.py` | CSI-ratio extraction, multi-ratio augmentation, SLSR samples |
| `wizecsi_dataset.py` | SLSR dataset with link mixing + link/ratio views |
| `wizecsi_models.py` | the network (One-For-All Net, modified AlexNet) |
| `wizecsi_utils.py` | training loop and late-fusion (Eq. 11) evaluation |
| `wizecsi_train.py` | training entry point and single-link validation |

## Citation

```bibtex
@article{dai2026wizecsi,
  author  = {Dai, Tonghua and Khorov, Evgeny},
  journal = {IEEE Access},
  title   = {{WiZeCSi}: Toward {Wi-Fi}-Based Single-Link Zero-Effort
             Cross-Domain Gesture Recognition},
  year    = {2026},
  volume  = {14},
  doi     = {10.1109/ACCESS.2026.3677814}
}
```
