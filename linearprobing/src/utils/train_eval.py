from typing import Optional
import sys
from pathlib import Path
import pickle
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from typing import Iterable, Union
import src.retfound.lr_sched as lr_sched
from torchvision.utils import save_image
#import matplotlib.pyplot as plt


import torch


class EarlyStopping:
    def __init__(
        self,
        patience=50,
        delta=0.01,
        greater_is_better=False,
        delta_two=0.01,
        greater_is_better_two=False,
        start_from=0
    ):
        self.patience = patience
        self.delta = delta
        self.delta_two = delta_two
        self.counter = 0
        self.best_value = None
        self.early_stop = False
        self.greater_is_better = greater_is_better
        if greater_is_better:
             self.is_better = lambda x, y: (x - y) > self.delta
        else:
            self.is_better = lambda x, y: (y - x) > self.delta
        if greater_is_better_two:
            self.is_better_two = lambda x, y: (x - y) > self.delta_two
        else:
            self.is_better_two = lambda x, y: (y - x) > self.delta_two
        self.is_same = lambda x, y: abs(x - y) < self.delta
        self.start_from = start_from

    def __call__(self, value, value_two, epoch):
        '''Returns True if the value is the best one so far, False
        otherwise.
        '''
        if (
            self.best_value is None
            # or (
            #     self.is_better(value, self.best_value)
            #     and self.is_better_two(value_two, self.best_value_two)
            # )
            or self.is_better(value, self.best_value)
            or ( # first value is the same, but second is better
                self.is_same(value, self.best_value)
                and self.is_better_two(value_two, self.best_value_two)
            )
        ):
            self.best_value = value
            self.best_value_two = value_two
            self.counter = 0
            return True
        elif epoch >= self.start_from:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return False


def train_1_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    args=None,
):
    assert args is not None, "args must be provided"

    model.train(True)
    optimizer.zero_grad()

    losses, true_labels, predictions = [], [], []
    for i, batch in enumerate(data_loader):
        images, targets = batch[0], batch[-1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # if i == 0:
        #     save_image(images, "images.png", normalize=True)

        # we use a per iteration (instead of per epoch) lr scheduler
        if i % args.accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, i / len(data_loader) + epoch, args)

        with torch.cuda.amp.autocast():
            # print(args.output_dir)
            # format: 3 numbers always
            if i == 0 and epoch % 10 == 0:
                epoch_str = str(epoch).zfill(3)
                save_fn = Path(args.output_dir, "debug", f"{epoch_str}_train.jpg")
                save_fn.parent.mkdir(exist_ok=True)
                print('Saving images for debugging')
                print('  images', images.shape, images.min().item(), images.max().item())
                print('  targets', targets.shape, targets.min().item(), targets.max().item())
                save_image(images, save_fn, normalize=True)
            # Make predictions for this batch
            outputs = model(images)

            # Compute the loss and its gradients
            loss = criterion(outputs, targets)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        # if loss in inf stop
        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            raise ValueError('Loss is infinite or NaN')
            # sys.exit(1)

        # reset gradients every accum_iter steps
        if (i + 1) % args.accum_iter == 0:
            optimizer.zero_grad()

        # adjust lrs
        min_lr, max_lr = 10.0, 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        # save true and predicted labels
        prediction_softmax = nn.Softmax(dim=1)(outputs)
        _, prediction_decode = torch.max(prediction_softmax, 1)

        predictions.extend(prediction_decode.cpu().detach().numpy())
        true_labels.extend(targets.cpu().detach().numpy())

    # gather the stats from all processes
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # compute avg loss over batches and Bacc
    avg_loss = np.mean(losses)
    bacc = balanced_accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average="weighted")

    if epoch % 5 == 0:
        print(
            f"[Train] Epoch {epoch} - Loss: {avg_loss:.4f}, Bacc: {bacc:.4f}, F1-score: {f1:.4f}"
        )

    return [epoch, avg_loss, bacc, f1]


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: Iterable,
    epoch: Union[int, str],
    device: torch.device,
    num_class: int,
    mode: str,
    get_embeddings: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    args=None,
    save_predictions: bool = False,
) -> Optional[list]:
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    prediction_decode_list = []
    prediction_list = []
    true_label_decode_list = []
    true_label_onehot_list = []

    # switch to evaluation mode
    model.eval()

    all_embeddings = {
        'embeddings': None,
        'targets': None
    }
    # Get dataloader length
    data_loader_len = len(data_loader)  # type: ignore
    period = max(int(round(0.1 * data_loader_len)), 1)
    # assert period > 0
    for bi, batch in enumerate(data_loader):
        images = batch[0]
        targets = batch[-1]
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        true_label = F.one_hot(targets.to(torch.int64), num_classes=num_class)

        # compute output
        # TODO: Do we really want to use mixed precision here?
        with torch.cuda.amp.autocast():
            if (
                (
                    isinstance(epoch, int)
                    and bi == 0
                    and epoch % 10 == 0
                    and args is not None
                 )
                or (
                    get_embeddings
                    and bi % period == 0
                )
            ):
                epoch_str = str(epoch).zfill(3)
                if get_embeddings:
                    assert save_path is not None
                    output_dir = Path(save_path).parent
                else:
                    assert args is not None
                    output_dir = args.output_dir
                save_fn = Path(output_dir, "debug", f"{epoch_str}_{mode}_{bi}.jpg")
                save_fn.parent.mkdir(exist_ok=True)
                print('Saving images for debugging')
                print('  images', images.shape, images.min().item(), images.max().item())
                print('  targets', targets.shape, targets.min().item(), targets.max().item())
                save_image(images, save_fn, normalize=True)
            if get_embeddings:
                # embeddings = model(images, get_embeddings=True)
                # embeddings = model.features(images)
                try:
                    embeddings = model.forward_features(images)
                except AttributeError:
                    embeddings = model(images)
                if save_path is not None:
                    for k, v in [('embeddings', embeddings), ('targets', targets)]:
                        if all_embeddings[k] is None:
                            all_embeddings[k] = v.cpu().detach().numpy()
                        else:
                            all_embeddings[k] = np.concatenate(
                                [all_embeddings[k], v.cpu().detach().numpy()]
                            )
                    print(all_embeddings['embeddings'].shape, all_embeddings['targets'].shape)
                    # with open(save_fn, "wb") as f:
                    #     pickle.dump(
                    #         dict(
                    #             embeddings=embeddings.cpu().detach().numpy(),
                    #             targets=targets.cpu().detach().numpy(),
                    #         ),
                    #         f,
                    #     )
                continue
            output = model(images)
            loss = criterion(output, targets)
            losses.append(loss.item())

            prediction_softmax = nn.Softmax(dim=1)(output)
            _, prediction_decode = torch.max(prediction_softmax, 1)
            _, true_label_decode = torch.max(true_label, 1)

            prediction_decode_list.extend(prediction_decode.cpu().detach().numpy())
            true_label_decode_list.extend(true_label_decode.cpu().detach().numpy())
            true_label_onehot_list.extend(true_label.cpu().detach().numpy())
            prediction_list.extend(prediction_softmax.cpu().detach().numpy())

    if get_embeddings and save_path is not None:
        assert all_embeddings['embeddings'] is not None
        assert all_embeddings['targets'] is not None
        save_fn = Path(save_path, "embeddings_targets.npz")
        np.savez_compressed(
            save_fn,
            embeddings=all_embeddings['embeddings'],
            targets=all_embeddings['targets']
        )
        # with open(save_fn, "wb") as f:
        #     pickle.dump(all_embeddings, f)
        return

    # if get_embeddings:
    #     return

    # gather the stats from all processes
    true_label_decode_list = np.array(true_label_decode_list)
    prediction_decode_list = np.array(prediction_decode_list)

    if save_predictions:
        print("Saving predictions")
        assert save_path is not None, "save_path must be provided"
        save_fn = Path(save_path, "predictions.npz")
        np.savez_compressed(
            save_fn,
            true_label_decode_list=true_label_decode_list,
            prediction_decode_list=prediction_decode_list,
            true_label_onehot_list=true_label_onehot_list,
            prediction_list=prediction_list,
        )
        #return

    avg_loss = np.mean(losses)
    acc = balanced_accuracy_score(true_label_decode_list, prediction_decode_list)
    auc_roc = roc_auc_score(
        true_label_onehot_list,
        prediction_list,
        multi_class="ovr",
        average="weighted",
    )
    auc_pr = average_precision_score(
        true_label_onehot_list, prediction_list, average="weighted"
    )
    f1 = f1_score(
        true_label_decode_list,
        prediction_decode_list,
        average="weighted",
        zero_division=0,
    )
    mcc = matthews_corrcoef(true_label_decode_list, prediction_decode_list)

    if type(epoch) != str and epoch % 5 == 0:
        print(
            "[{}] Epoch {} - Loss: {:.4f}, Bacc: {:.4f} AUROC: {:.4f} AP: {:.4f} F1-score: {:.4f} MCC: {:.4f}".format(
                mode, epoch, avg_loss, acc, auc_roc, auc_pr, f1, mcc
            )
        )

    return [epoch, avg_loss, acc, auc_roc, auc_pr, f1, mcc]
