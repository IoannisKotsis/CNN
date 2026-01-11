import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import copy
import datetime
from pathlib import Path
import numpy as np


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, train_loader, optimizer, criteria, device):
    criterion_single, criterion_multi, criterion_binary = criteria

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, social_labels, creator_labels, logo_labels in train_loader:
        images = images.to(device)
        social_labels = social_labels.to(device)
        creator_labels = creator_labels.to(device)
        logo_labels = logo_labels.to(device)

        optimizer.zero_grad()
        social_logits, creator_logits, logo_logits = model(images)

        loss1 = criterion_single(social_logits, social_labels)
        loss2 = criterion_multi(creator_logits, creator_labels)
        loss3 = criterion_binary(logo_logits, logo_labels)
        loss = loss1 + loss2 + loss3

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = social_logits.argmax(1)
        correct += (preds == social_labels).sum().item()
        total += images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    acc_single = (correct / total) * 100
    return epoch_loss, acc_single


@torch.no_grad()
def validate(model, val_loader, criteria, device, creator_num_labels, multilabel_threshold):
    criterion_single, criterion_multi, criterion_binary = criteria

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    TP = torch.zeros(creator_num_labels, dtype=torch.long)
    TN = torch.zeros(creator_num_labels, dtype=torch.long)
    FP = torch.zeros(creator_num_labels, dtype=torch.long)
    FN = torch.zeros(creator_num_labels, dtype=torch.long)

    for images, social_labels, creator_labels, logo_labels in val_loader:
        images = images.to(device)
        social_labels = social_labels.to(device)
        creator_labels = creator_labels.to(device)
        logo_labels = logo_labels.to(device)

        social_logits, creator_logits, logo_logits = model(images)

        loss1 = criterion_single(social_logits, social_labels)
        loss2 = criterion_multi(creator_logits, creator_labels)
        loss3 = criterion_binary(logo_logits, logo_labels)
        loss = loss1 + loss2 + loss3
        val_loss += loss.item() * images.size(0)

        probs = torch.sigmoid(creator_logits)
        preds = (probs > multilabel_threshold).float()

        TP += ((preds == 1) & (creator_labels == 1)).sum(dim=0).cpu()
        TN += ((preds == 0) & (creator_labels == 0)).sum(dim=0).cpu()
        FP += ((preds == 1) & (creator_labels == 0)).sum(dim=0).cpu()
        FN += ((preds == 0) & (creator_labels == 1)).sum(dim=0).cpu()

        social_preds = social_logits.argmax(1)
        correct += (social_preds == social_labels).sum().item()
        total += images.size(0)

    final_val_loss = val_loss / len(val_loader.dataset)

    denom = TP + TN + FP + FN
    denom = torch.clamp(denom, min=1)  # avoid division by 0
    per_label_acc = (TP + TN) / denom
    macro_multi_acc = per_label_acc.mean().item()
    single_acc = (correct / total) * 100

    return final_val_loss, macro_multi_acc, single_acc


@torch.no_grad()
def test(model, test_loader, criteria, device, label_maps,
         multilabel_threshold, binary_threshold):
    criterion_single, criterion_multi, criterion_binary = criteria

    model.eval()
    test_loss = 0.0

    all_single = []
    all_single_preds = []
    all_multi = []
    all_multi_preds = []
    all_binary = []
    all_binary_preds = []

    creator_num_labels = len(label_maps["creator"])

    TP = torch.zeros(creator_num_labels, dtype=torch.long)
    TN = torch.zeros(creator_num_labels, dtype=torch.long)
    FP = torch.zeros(creator_num_labels, dtype=torch.long)
    FN = torch.zeros(creator_num_labels, dtype=torch.long)

    correct = 0
    total = 0

    for images, social_labels, creator_labels, logo_labels in test_loader:
        images = images.to(device)
        social_labels = social_labels.to(device)
        creator_labels = creator_labels.to(device)
        logo_labels = logo_labels.to(device)

        social_logits, creator_logits, logo_logits = model(images)

        loss1 = criterion_single(social_logits, social_labels)
        loss2 = criterion_multi(creator_logits, creator_labels)
        loss3 = criterion_binary(logo_logits, logo_labels)
        loss = loss1 + loss2 + loss3
        test_loss += loss.item() * images.size(0)

        # single-label
        social_preds = social_logits.argmax(1)
        correct += (social_preds == social_labels).sum().item()
        total += images.size(0)

        # multi-label
        probs = torch.sigmoid(creator_logits)
        multi_preds = (probs > multilabel_threshold).int()

        TP += ((multi_preds == 1) & (creator_labels == 1)).sum(dim=0).cpu()
        TN += ((multi_preds == 0) & (creator_labels == 0)).sum(dim=0).cpu()
        FP += ((multi_preds == 1) & (creator_labels == 0)).sum(dim=0).cpu()
        FN += ((multi_preds == 0) & (creator_labels == 1)).sum(dim=0).cpu()

        # binary
        logo_probs = torch.sigmoid(logo_logits)
        logo_preds = (logo_probs > binary_threshold).int()

        all_binary_preds.extend(logo_preds.view(-1).cpu().numpy().astype(int))
        all_binary.extend(logo_labels.view(-1).cpu().numpy().astype(int))

        all_single.extend(social_labels.cpu().numpy().astype(int))
        all_single_preds.extend(social_preds.cpu().numpy().astype(int))
        all_multi.extend(creator_labels.cpu().numpy().astype(int))
        all_multi_preds.extend(multi_preds.cpu().numpy().astype(int))

    final_test_loss = test_loss / len(test_loader.dataset)
    single_acc = (correct / total) * 100

    testing_macro_f1 = f1_score(all_multi, all_multi_preds, average="macro", zero_division=0)
    testing_micro_f1 = f1_score(all_multi, all_multi_preds, average="micro", zero_division=0)

    binary_f1 = f1_score(all_binary, all_binary_preds, average=None, zero_division=0)

    conf_matrix = confusion_matrix(
        all_single, all_single_preds, labels=np.arange(len(label_maps["social"]))
    )

    return {
        "test_loss": final_test_loss,
        "single_acc": single_acc,
        "multi_macro_f1": testing_macro_f1,
        "multi_micro_f1": testing_micro_f1,
        "binary_f1": binary_f1,
        "confusion_matrix": conf_matrix,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN
    }


def fit(model, train_loader, val_loader,
        epochs, lr,
        min_delta, patience,
        creator_num_labels,
        val_multilabel_threshold,
        checkpoint_path,
        log_dir="runs/tb_logs"):
    device = get_device()
    model.to(device)

    criteria = (
        nn.CrossEntropyLoss(),
        nn.BCEWithLogitsLoss(),
        nn.BCEWithLogitsLoss()
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # tensorboard
    day_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    time_stamp = datetime.datetime.now().strftime("%H-%M-%S")
    run_path = Path(log_dir) / day_stamp / time_stamp
    run_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_path)

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criteria, device)

        val_loss, val_macro_multi_acc, val_single_acc = validate(
            model, val_loader, criteria, device, creator_num_labels, val_multilabel_threshold
        )

        writer.add_scalars("Loss Curves", {"Training Loss": train_loss, "Validation Loss": val_loss}, epoch)
        writer.add_scalars("Accuracy Metrics", {"Validation Multi (macro acc)": val_macro_multi_acc}, epoch)

        if (best_val_loss - val_loss) > min_delta:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(), "loss": train_loss},
                checkpoint_path
            )
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}, best val loss: {best_val_loss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    writer.close()
    return model, device, criteria
