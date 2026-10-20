"""WiZeCSi training and evaluation loops.

Two different accuracies are reported:

* **sample level** -- the accuracy of a single SLSR sample, the quantity the
  network is trained on;
* **instance level** -- the accuracy of a gesture, obtained by averaging the
  logits of all SLSR samples of that gesture instance (late fusion, Eq. 11).
  This is the number the paper reports.
"""

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, tensor_loader, num_epochs, learning_rate, l2_lambda, criterion,
          device, scheduler=(8, 0.1), eval_fn=None):
    """Train ``model`` and return ``(train_log, val_log)`` of shape ``(2, epochs)``.

    Row 0 is the accuracy, row 1 the loss.  When ``eval_fn(model) -> float`` is
    given the model is evaluated after every epoch.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=l2_lambda)
    scheduler = StepLR(optimizer, step_size=scheduler[0], gamma=scheduler[1])
    train_log = np.zeros((2, num_epochs))
    val_log = np.zeros((2, num_epochs))

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_acc = 0.0, 0.0
        bar = tqdm(tensor_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", mininterval=1)
        for inputs, labels in bar:
            inputs = inputs.to(device, non_blocking=True).float()
            labels = labels.to(device, non_blocking=True).long()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_acc += (outputs.argmax(dim=1) == labels).sum().item()
        scheduler.step()

        n_seen = max(len(tensor_loader.dataset), 1)
        train_log[0][epoch] = running_acc / n_seen
        train_log[1][epoch] = running_loss / n_seen
        print(f"Epoch {epoch + 1}: accuracy {train_log[0][epoch]:.4f}, "
              f"loss {train_log[1][epoch]:.9f}")

        if eval_fn is not None:
            val_log[0][epoch] = eval_fn(model)
            print(f"  validation instance accuracy: {val_log[0][epoch]:.4f}")
    return train_log, val_log


@torch.no_grad()
def sample_accuracy(model, tensor_loader, criterion, device):
    """Mean accuracy and loss over individual SLSR samples."""
    model.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    for inputs, labels in tensor_loader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        outputs = model(inputs)
        total_loss += criterion(outputs, labels).item() * inputs.size(0)
        total_acc += (outputs.argmax(dim=1) == labels).sum().item()
        n_batches += 1
    n = max(len(tensor_loader.dataset), 1)
    return total_acc / n, total_loss / n


@torch.no_grad()
def fused_accuracy(model, dataset, device, batch_size=240, num_workers=0, verbose=False):
    """Instance accuracy with late fusion over all SLSR samples (Eq. 11).

    The logits of every SLSR sample of the same gesture instance are averaged
    before the softmax, so the prediction does not depend on how many links or
    ratios happen to be available.
    """
    if isinstance(dataset, DataLoader):
        loader, dataset = dataset, dataset.dataset
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    model.eval()

    keys, inverse = np.unique(np.asarray(dataset.inst_list), return_inverse=True)
    keys = list(keys)
    if len(keys) == 0:
        raise ValueError("the evaluation dataset contains no instance")
    labels = np.zeros(len(keys), dtype=np.int64)
    labels[inverse] = np.asarray(dataset.label_list)

    sums, pos = None, 0
    for inputs, _ in tqdm(loader, desc="evaluating", mininterval=1, disable=not verbose):
        logits = model(inputs.to(device).float()).float().cpu().numpy()
        if sums is None:
            sums = np.zeros((len(keys), logits.shape[1]), dtype=np.float64)
        np.add.at(sums, inverse[pos:pos + len(logits)], logits)
        pos += len(logits)

    predictions = sums.argmax(axis=1)
    return float((predictions == labels).mean())
