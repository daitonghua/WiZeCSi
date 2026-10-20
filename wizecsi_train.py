"""WiZeCSi: train on Widar3 and validate with single-link late fusion.

The model is trained on all links and all CSI ratios at once (*link mixing* +
*multi-ratio augmentation*), then validated on the held-out repetition with one
single link at a time.  Each link is predicted from its own CSI ratios only,
and the logits are fused per gesture instance (Eq. 11), so the reported number
is the single-link accuracy the paper is about.
"""

import argparse
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wizecsi_dataset import ALL_CLASS, ALL_LINK, ALL_RATIO, SLSRDataset, SLSRView
from wizecsi_models import OneForAllNet
from wizecsi_utils import fused_accuracy, sample_accuracy
from wizecsi_utils import train as train_loop

CFG = dict(
    data_root=os.path.join("data", "widar3_slsr"),
    classes=ALL_CLASS,
    train_users=[str(i) for i in range(11, 17)],
    test_users=None,                # None -> in-domain, same users as training
    train_reps=["1", "2", "3", "4"],
    test_reps=["5"],
    locs=[str(i) for i in range(1, 6)],
    oris=[str(i) for i in range(1, 6)],
    links=ALL_LINK,                 # links used for training (link mixing)
    ratios=ALL_RATIO,               # ratios used for training (augmentation)
    n_sz=2, n_emb=128, p_dropout=0.5,
    lr=3e-3, bs=240, wd=1e-5,
    step_size=8, gamma=0.1,
    epochs=20,
)


def build_sets(cfg):
    """The SLSR training set and the held-out instance set."""
    train_set = SLSRDataset(
        root_dir=cfg["data_root"], class_sel=cfg["classes"], users=cfg["train_users"],
        locs=cfg["locs"], oris=cfg["oris"], links=cfg["links"], reps=cfg["train_reps"],
        ratios=cfg["ratios"])

    test_users = cfg["test_users"] or cfg["train_users"]
    test_set = SLSRDataset(
        root_dir=cfg["data_root"], class_sel=cfg["classes"], users=test_users,
        locs=cfg["locs"], oris=cfg["oris"], links=cfg["links"], reps=cfg["test_reps"],
        ratios=cfg["ratios"])

    if len(train_set) == 0 or len(test_set) == 0:
        empty = "training" if len(train_set) == 0 else "test"
        raise SystemExit(
            f"the {empty} set is empty -- check --users/--locs/--oris/--links/--ratios "
            f"and --train-reps/--test-reps against what wizecsi_preprocess.py generated "
            f"in {cfg['data_root']}")
    return train_set, test_set


def evaluate(model, test_set, cfg, device):
    """Single-link accuracy for every link, then all links fused together."""
    print("\n--- validation (late fusion over the ratios of one link) ---")
    accs = {}
    for link in cfg["links"]:
        view = SLSRView(test_set, links=[link], ratios=cfg["ratios"])
        if len(view) == 0:
            continue
        accs[link] = fused_accuracy(model, view, device, batch_size=cfg["bs"])
        print(f"  link {link}: {accs[link] * 100:.2f}%  ({len(view)} samples)")

    all_links = fused_accuracy(model, test_set, device, batch_size=cfg["bs"])
    mean_link = float(np.mean(list(accs.values()))) if accs else float("nan")
    print(f"  mean over single links: {mean_link * 100:.2f}%")
    print(f"  all {len(cfg['links'])} links fused: {all_links * 100:.2f}%")
    return accs, mean_link, all_links


def main():
    ap = argparse.ArgumentParser(description="WiZeCSi: training and validation on Widar3")
    ap.add_argument("--data-root", default=CFG["data_root"],
                    help="root of the SLSR dataset built by wizecsi_preprocess.py")
    ap.add_argument("--classes", type=int, nargs="+", default=CFG["classes"])
    ap.add_argument("--users", nargs="+", default=CFG["train_users"], help="training users")
    ap.add_argument("--test-users", nargs="+", default=CFG["test_users"],
                    help="held-out users (default: same as --users, i.e. in-domain)")
    ap.add_argument("--locs", nargs="+", default=CFG["locs"])
    ap.add_argument("--oris", nargs="+", default=CFG["oris"])
    ap.add_argument("--links", nargs="+", default=CFG["links"],
                    help="links mixed in training; all of them are validated")
    ap.add_argument("--ratios", nargs="+", default=CFG["ratios"],
                    help="CSI ratios used by multi-ratio augmentation")
    ap.add_argument("--train-reps", nargs="+", default=CFG["train_reps"])
    ap.add_argument("--test-reps", nargs="+", default=CFG["test_reps"])
    ap.add_argument("--epochs", type=int, default=CFG["epochs"])
    ap.add_argument("--lr", type=float, default=CFG["lr"])
    ap.add_argument("--bs", type=int, default=CFG["bs"])
    ap.add_argument("--wd", type=float, default=CFG["wd"])
    ap.add_argument("--n-sz", type=int, default=CFG["n_sz"], help="channel width multiplier")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-root", default=os.path.join("save_files", "widar3"))
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = dict(CFG)
    cfg.update(data_root=args.data_root, classes=args.classes, train_users=args.users,
               test_users=args.test_users, locs=args.locs, oris=args.oris,
               links=args.links, ratios=args.ratios, train_reps=args.train_reps,
               test_reps=args.test_reps, epochs=args.epochs, lr=args.lr, bs=args.bs,
               wd=args.wd, n_sz=args.n_sz)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_set, test_set = build_sets(cfg)
    print(f"train: {train_set}")
    print(f"test : {test_set}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OneForAllNet(len(cfg["classes"]), n_sz=cfg["n_sz"],
                         p_dropout=CFG["p_dropout"], n_emb=CFG["n_emb"]).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"device: {device}   model params: {n_par / 1e6:.2f} M")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=cfg["bs"], shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin,
                              drop_last=False)

    # cheap per-epoch probe: one link, its ratios fused per instance
    probe = SLSRView(test_set, links=cfg["links"][:1])

    def eval_fn(m):
        return fused_accuracy(m, probe, device, batch_size=cfg["bs"])

    log = train_loop(model=model, tensor_loader=train_loader, num_epochs=cfg["epochs"],
                     learning_rate=cfg["lr"], l2_lambda=cfg["wd"],
                     criterion=nn.CrossEntropyLoss(), device=device,
                     scheduler=(cfg["step_size"], cfg["gamma"]), eval_fn=eval_fn)

    with torch.no_grad():
        test_loader = DataLoader(test_set, batch_size=cfg["bs"], shuffle=False,
                                 num_workers=args.num_workers, pin_memory=pin)
        sample_acc, sample_loss = sample_accuracy(model, test_loader,
                                                  nn.CrossEntropyLoss(), device)
    print(f"\ntest sample accuracy: {sample_acc * 100:.2f}%   loss: {sample_loss:.5f}")

    accs, mean_link, all_links = evaluate(model, test_set, cfg, device)

    name = (f"wizecsi_widar3_classroom_links{''.join(cfg['links'])}_"
            f"ratios{''.join(cfg['ratios'])}_train_r{''.join(cfg['train_reps'])}_"
            f"test_r{''.join(cfg['test_reps'])}_ep{cfg['epochs']}_"
            f"singlelink{mean_link * 100:.2f}_alllink{all_links * 100:.2f}")
    os.makedirs(args.save_root, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_root, name + ".pt"))
    with open(os.path.join(args.save_root, "log_" + name + ".pkl"), "wb") as f:
        pickle.dump({"train": log, "per_link": accs,
                     "single_link": mean_link, "all_links": all_links}, f)
    print(f"saved: {name}.pt")


if __name__ == "__main__":
    main()
