"""WiZeCSi model: the One-For-All Net of Fig. 3.

A deliberately small modified AlexNet.  It sees one SLSR sample
``(1, T_MAX, 2F) = (1, 3000, 60)`` -- the amplitude and the phase of one CSI
ratio -- and knows nothing about the link or the ratio it came from, which is
what makes link mixing and cross-link sensing possible.  A LayerNorm is applied
before the linear output layer so that the embedding is well conditioned.

With the paper's width (``n_sz = 2``) the network has 20.6 M parameters and costs
1.21 GFLOPs per SLSR sample, matching Section V-F.  ``n_sz = 1`` is a half-width
variant for quick experiments.
"""

import torch
import torch.nn as nn

IN_ROWS = 3000          # T_MAX packets
IN_COLS = 60            # 2F: 30 amplitude + 30 phase subcarriers
FEAT_ROWS = 36 * 4      # spatial size after the convolutional stack
N_SZ = 2                # channel width multiplier used in the paper


class FeatureExtractor(nn.Module):
    """5 conv layers, 2 max pools, 1 average pool, 2 FC layers, LayerNorm."""

    def __init__(self, p_dropout=0.5, n_sz=N_SZ, n_emb=128):
        super().__init__()
        self.features = nn.Sequential(
            # (1, 3000, 60) -> (32 n_sz, 599, 27)
            nn.Conv2d(1, 32 * n_sz, 11, stride=(5, 2), padding=2),
            nn.BatchNorm2d(32 * n_sz),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),                      # -> 299, 13

            nn.Conv2d(32 * n_sz, 64 * n_sz, 5, stride=(2, 1), padding=2),
            nn.BatchNorm2d(64 * n_sz),
            nn.ReLU(True),
            nn.MaxPool2d(3, stride=2),                      # -> 74, 6

            nn.Conv2d(64 * n_sz, 128 * n_sz, 3, stride=1, padding=1),
            nn.BatchNorm2d(128 * n_sz),
            nn.ReLU(True),
            nn.Conv2d(128 * n_sz, 128 * n_sz, 3, stride=1, padding=1),
            nn.BatchNorm2d(128 * n_sz),
            nn.ReLU(True),
            nn.Conv2d(128 * n_sz, 128 * n_sz, 3, stride=1, padding=1),
            nn.BatchNorm2d(128 * n_sz),
            nn.ReLU(True),
            nn.AvgPool2d(3, stride=(2, 1)),                 # -> 36, 4
        )
        self.n_feat = 128 * n_sz * FEAT_ROWS
        self.fc = nn.Sequential(
            nn.Dropout(p=p_dropout),
            nn.Linear(self.n_feat, n_emb * 4),
            nn.ReLU(True),

            nn.Dropout(p=0.5),
            nn.Linear(n_emb * 4, n_emb),
            nn.ReLU(True),

            nn.LayerNorm(n_emb, elementwise_affine=False),
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.flatten(1))


class OneForAllNet(nn.Module):
    """The One-For-All Net: one model for every link and every CSI ratio."""

    def __init__(self, num_classes, n_sz=N_SZ, p_dropout=0.5, n_emb=128):
        super().__init__()
        self.fe = FeatureExtractor(p_dropout, n_sz=n_sz, n_emb=n_emb)
        self.clf = nn.Linear(n_emb, num_classes)

    def forward(self, x):
        return self.clf(self.fe(x))


def count_macs(model):
    """Multiply-accumulates of one forward pass, counted as the papers do."""
    macs = [0]

    def hook(module, _, out):
        if isinstance(module, nn.Conv2d):
            macs[0] += out.numel() * module.weight[0].numel()
        elif isinstance(module, nn.Linear):
            macs[0] += out.numel() * module.in_features

    handles = [m.register_forward_hook(hook) for m in model.modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, IN_ROWS, IN_COLS))
    model.train(was_training)
    for handle in handles:
        handle.remove()
    return macs[0]


if __name__ == "__main__":
    net = OneForAllNet(6)
    n_par = sum(p.numel() for p in net.parameters())
    out = net(torch.randn(2, 1, IN_ROWS, IN_COLS))
    print(f"n_sz={N_SZ}   params: {n_par / 1e6:.2f} M   "
          f"MACs: {count_macs(net) / 1e9:.3f} G   output: {tuple(out.shape)}")
    print("paper (Sec. V-F): 1.21 GFLOPs per SLSR sample for the original model")

