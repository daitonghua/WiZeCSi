"""WiZeCSi dataset: the SLSR dataset produced by ``wizecsi_preprocess.py``.

Because of *multi-ratio augmentation* every measurement contributes ``M`` SLSR
samples, and because of *link mixing* all links live in one single dataset with
no link label.  One epoch therefore iterates over samples, not over gestures;
:class:`SLSRView` restricts that dataset to a subset of links or ratios, which
is what the single-link and arbitrary-link-combination tests need.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

ALL_LINK = [str(i) for i in range(1, 7)]    # six Widar3 receivers
ALL_RATIO = [str(i) for i in range(1, 7)]   # M = N(N - 1) = 6 CSI ratios
ALL_CLASS = [1, 2, 3, 4, 6, 9]
ALL_USER = [str(i) for i in range(11, 17)]
ALL_LOC = [str(i) for i in range(1, 6)]
ALL_ORI = [str(i) for i in range(1, 6)]
ALL_REP = [str(i) for i in range(1, 6)]


def load_sample(path, transform=None):
    """Read one ``(T_MAX, 60)`` SLSR sample as a ``(1, T_MAX, 60)`` tensor."""
    x = torch.from_numpy(np.asarray(np.load(path), dtype=np.float32))
    if transform is not None:
        x = transform(x)
    return x[None, :, :]


class SLSRDataset(Dataset):
    """Single-Link-Single-Ratio samples stored as ``<root>/<class>/<name>.npy``.

    A file name is ``user<u>-<label>-<loc>-<ori>-<rep>-r<rx>-<ratio>.npy``; the
    part before the last dash identifies the gesture instance the sample was
    augmented from.
    """

    def __init__(self, root_dir, class_sel=None, users=None, locs=None, oris=None,
                 links=None, reps=None, ratios=None, transforms=None):
        super().__init__()
        self.root = root_dir
        self.class_sel = list(ALL_CLASS if class_sel is None else class_sel)
        self.users = set(ALL_USER if users is None else users)
        self.locs = set(ALL_LOC if locs is None else locs)
        self.oris = set(ALL_ORI if oris is None else oris)
        self.links = set(ALL_LINK if links is None else links)
        self.reps = set(ALL_REP if reps is None else reps)
        self.ratios = set(ALL_RATIO if ratios is None else ratios)
        self.transform = transforms

        self.data_list = []    # absolute path of every SLSR sample
        self.label_list = []   # gesture class index
        self.link_list = []    # receiver id, kept only for slicing
        self.ratio_list = []   # CSI ratio id, kept only for slicing
        self.inst_list = []    # instance id, the late fusion unit (Eq. 11)
        self._index()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return load_sample(self.data_list[idx], self.transform), self.label_list[idx]

    def __repr__(self):
        return (f"{self.__class__.__name__}(samples={len(self)}, "
                f"instances={len(set(self.inst_list))}, "
                f"links={len(set(self.link_list))}, "
                f"ratios={len(set(self.ratio_list))})")

    def _index(self):
        for cl in self.class_sel:
            cl_dir = os.path.join(self.root, str(cl))
            if not os.path.isdir(cl_dir):
                raise FileNotFoundError(
                    f"missing class directory: {cl_dir}\n"
                    f"build it first with: wizecsi_preprocess.py --save-root {self.root}")
            label = self.class_sel.index(cl)
            for name in sorted(os.listdir(cl_dir)):
                if not name.endswith(".npy"):
                    continue
                parts = name[:-4].split("-")
                if len(parts) != 7:                 # not an SLSR sample
                    continue
                user, loc, ori, rep = parts[0][4:], parts[2], parts[3], parts[4]
                rx, ratio = parts[5][1:], parts[6]
                if (user not in self.users or loc not in self.locs
                        or ori not in self.oris or rep not in self.reps
                        or rx not in self.links or ratio not in self.ratios):
                    continue
                self.data_list.append(os.path.join(cl_dir, name))
                self.label_list.append(label)
                self.link_list.append(rx)
                self.ratio_list.append(ratio)
                self.inst_list.append(name[:-4].rsplit("-", 1)[0])


class SLSRView(Dataset):
    """A link- and/or ratio-restricted view over an :class:`SLSRDataset`.

    No re-scan of the filesystem: the view keeps the matching subset of the base
    sample lists, so a single-link test is just
    ``SLSRView(test_set, links=['3'])``.
    """

    def __init__(self, base, links=None, ratios=None):
        self.transform = base.transform
        self.data_list, self.label_list = [], []
        self.link_list, self.ratio_list, self.inst_list = [], [], []
        for i in range(len(base)):
            if links is not None and base.link_list[i] not in links:
                continue
            if ratios is not None and base.ratio_list[i] not in ratios:
                continue
            self.data_list.append(base.data_list[i])
            self.label_list.append(base.label_list[i])
            self.link_list.append(base.link_list[i])
            self.ratio_list.append(base.ratio_list[i])
            self.inst_list.append(base.inst_list[i])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return load_sample(self.data_list[idx], self.transform), self.label_list[idx]

    def __repr__(self):
        return (f"{self.__class__.__name__}(samples={len(self)}, "
                f"instances={len(set(self.inst_list))}, "
                f"links={len(set(self.link_list))}, "
                f"ratios={len(set(self.ratio_list))})")
