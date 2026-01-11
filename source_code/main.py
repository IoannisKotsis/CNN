# main.py

from config import (
    batch_size,
    epoch_number,
    lr,
    min_delta,
    resize_width,
    resize_height,
    train_split_pct,
    validation_split_pct,
    validation_multilabel_threshold,
    testing_multilabel_threshold,
    testing_binary_threshold,
)

from data_preparation import prepare_data
from dataset import ImageDataset
from model import build_model
from train_eval import fit, test

from torch.utils.data import DataLoader
from torchvision import transforms
import os


def main():
    # -------------------------
    # 1) Project folders
    # -------------------------
    project_dir = "project_ki"
    csv_dir = os.path.join(project_dir, "csv_files")
    checkpoints_dir = os.path.join(project_dir, "checkpoints")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # -------------------------
    # 2) Prepare data
    # -------------------------
    # ΒΑΛΕ ΕΔΩ ΤΑ PATHS ΣΟΥ
    JSON_PATH = "path/to/annotations.json"
    IMAGES_FOLDER = "path/to/images_folder"

    train_csv, val_csv, test_csv, label_maps = prepare_data(
        json_path=JSON_PATH,
        images_folder_path=IMAGES_FOLDER,
        csv_dir=csv_dir,
        train_split_pct=train_split_pct,
        validation_split_pct=validation_split_pct,
    )

    # -------------------------
    # 3) Transforms
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((resize_width, resize_height)),
        transforms.ToTensor(),
    ])

    # -------------------------
    # 4) Datasets & Loaders
    # -------------------------
    train_dataset = ImageDataset(
        train_csv,
        label_maps["social"],
        label_maps["creator"],
        label_maps["logo"],
        transform=transform,
    )

    val_dataset = ImageDataset(
        val_csv,
        label_maps["social"],
        label_maps["creator"],
        label_maps["logo"],
        transform=transform,
    )

    test_dataset = ImageDataset(
        test_csv,
        label_maps["social"],
        label_maps["creator"],
        label_maps["logo"],
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # -------------------------
    # 5) Model
    # -------------------------
    model = build_model(
        label_maps=label_maps,
        input_dims=(resize_width, resize_height, 3),
    )

    # -------------------------
    # 6) Training
    # -------------------------
    checkpoint_path = os.path.join(checkpoints_dir, "CNN_best_model.pth")

    model, device, criteria = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epoch_number,
        lr=lr,
        min_delta=min_delta,
        patience=5,
        creator_num_labels=len(label_maps["creator"]),
        val_multilabel_threshold=validation_multilabel_threshold,
        checkpoint_path=checkpoint_path,
    )

    # -------------------------
    # 7) Testing
    # -------------------------
    metrics = test(
        model=model,
        test_loader=test_loader,
        criteria=criteria,
        device=device,
        label_maps=label_maps,
        multilabel_threshold=testing_multilabel_threshold,
        binary_threshold=testing_binary_threshold,
    )

    print("\nFinal test metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
