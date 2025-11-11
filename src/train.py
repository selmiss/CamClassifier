import os
import argparse
import random
from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
from src.predict import list_images, crop_box


def build_model(num_classes: int) -> Tuple[nn.Module, object]:
    """
    Create MobileNetV3-Small with ImageNet weights and replace the classifier head.
    Returns the model and its default transforms.
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT
    preprocess = weights.transforms()
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model, preprocess


class FewShotDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], transform):
        self.items = items
        self.transform = transform
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        # Use the same crop area as inference for consistency
        # img = crop_box(img, left_ratio=0.0, top_ratio=0.2, right_ratio=0.5, bottom_ratio=0.95)
        x = self.transform(img)
        return x, label


def train_head_only(model: nn.Module,
                    loader: DataLoader,
                    device: torch.device,
                    class_counts: List[int],
                    epochs: int = 10,
                    lr: float = 1e-3,
                    weight_decay: float = 1e-4) -> None:
    """
    Fine-tune only the classifier head with frozen backbone.
    """
    # Freeze backbone - only train the classifier
    for p in model.features.parameters():
        p.requires_grad = False
    
    # Unfreeze classifier head
    for p in model.classifier.parameters():
        p.requires_grad = True
    
    model.to(device).train()

    class_weights = torch.tensor([1.0 / max(c, 1) for c in class_counts],
                                 dtype=torch.float32, device=device)
    # Only optimize classifier parameters
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(epochs):
        running_loss = 0.0
        num_samples = 0
        correct = 0
        max_batch_loss = 0.0
        batches = 0
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Logging
            batch_size = yb.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size
            correct += (logits.argmax(dim=-1) == yb).sum().item()
            max_batch_loss = max(max_batch_loss, loss.item())
            batches += 1
        if num_samples > 0:
            avg_loss = running_loss / num_samples
            train_acc = correct / num_samples
            print(
                f"Epoch {epoch+1:02}/{epochs:02} | "
                f"loss={avg_loss:.4f} | "
                f"acc={train_acc:.2%} | "
                f"max_loss={max_batch_loss:.4f} | "
                f"batches={batches}"
            )


def main():
    parser = argparse.ArgumentParser(description="Few-shot fine-tuning for door state classifier")
    parser.add_argument("--fewshot_dir", type=str, default="logs/data/training_data",
                        help="Root dir containing pos/, neg/, neg_dark/ (use '@test_data' to map to logs/data/test_data)")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for fine-tuning")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-2, help="Learning rate for head (backbone uses lr/10)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--test_split", type=float, default=0, help="Ratio of data for test set (default 0.3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Where to save checkpoint")
    parser.add_argument("--checkpoint_name", type=str, default="mobilenet_v3_small_11_11_2025.pt",
                        help="Checkpoint file name")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Allow alias to use the attached test_data as training source
    if args.fewshot_dir == "@test_data":
        args.fewshot_dir = os.path.join("logs", "data", "test_data")

    # Define two classes; map both neg and neg_dark into "closed"
    classes = ["open", "closed"]
    class_to_subdirs = {
        "open": ["pos"],
        "closed": ["neg", "neg_dark"],
    }

    # Gather samples and split into train/test
    train_samples: List[Tuple[str, int]] = []
    test_samples: List[Tuple[str, int]] = []
    train_counts = [0] * len(classes)
    test_counts = [0] * len(classes)
    
    print(f"Splitting data: test_ratio={args.test_split}, seed={args.seed}\n")
    for idx, cname in enumerate(classes):
        class_paths = []
        for sub in class_to_subdirs[cname]:
            folder = os.path.join(args.fewshot_dir, sub)
            paths = list_images(folder)
            if paths:
                class_paths.extend(paths)
        
        if not class_paths:
            raise ValueError(f"No images found for class '{cname}' in subdirs: {class_to_subdirs[cname]}")
        
        # Shuffle and split
        random.shuffle(class_paths)
        n_test = int(len(class_paths) * args.test_split)
        test_paths = class_paths[:n_test]
        train_paths = class_paths[n_test:]
        
        print(f"  {cname}: {len(class_paths)} total -> {len(train_paths)} train, {len(test_paths)} test")
        
        train_samples.extend([(p, idx) for p in train_paths])
        test_samples.extend([(p, idx) for p in test_paths])
        train_counts[idx] = len(train_paths)
        test_counts[idx] = len(test_paths)

    # Build model and dataloaders
    model, preprocess = build_model(num_classes=len(classes))
    train_dataset = FewShotDataset(train_samples, preprocess)
    test_dataset = FewShotDataset(train_samples, preprocess)  # Use same samples as training
    
    bs = min(args.batch_size, max(4, len(train_dataset)))
    loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # Train model
    print(f"\nTraining for {args.epochs} epoch(s) on device {device}...")
    train_head_only(model, loader, device, class_counts=train_counts,
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
    model.eval()

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    
    test_acc = correct / total if total > 0 else 0.0
    print(f"  Test Accuracy: {test_acc:.2%} ({correct}/{total})")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    torch.save({
        "model_state": model.state_dict(),
        "classes": classes,
        "test_accuracy": test_acc,
    }, ckpt_path)
    print(f"\n✓ Checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()


