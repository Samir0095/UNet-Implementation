import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from model import UNet
from dataloader import get_dataloaders


# ============================================================
# Configuration
# ============================================================

DATASET_PATH = (
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet\Datasets\BraTS2018"
)

BATCH_SIZE = 1

NUM_EPOCHS = 2

LEARNING_RATE = 1e-4

MODEL_SAVE_PATH = "best_model.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nDevice:")
print(device)

if torch.cuda.is_available():

    print("GPU:")
    print(torch.cuda.get_device_name(0))


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(-1)

        targets = targets.view(-1)

        intersection = (
            predictions * targets
        ).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            predictions.sum()
            + targets.sum()
            + self.smooth
        )

        return 1.0 - dice


# ============================================================
# Combined Loss
# ============================================================

class DiceBCELoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, predictions, targets):

        bce_loss = self.bce(
            predictions,
            targets
        )

        dice_loss = self.dice(
            predictions,
            targets
        )

        return bce_loss + dice_loss


# ============================================================
# Dice Score
# ============================================================

def dice_score(predictions, targets, threshold=0.5):

    predictions = torch.sigmoid(predictions)

    predictions = (
        predictions > threshold
    ).float()

    predictions = predictions.view(-1)

    targets = targets.view(-1)

    intersection = (
        predictions * targets
    ).sum()

    dice = (
        2.0 * intersection + 1.0
    ) / (
        predictions.sum()
        + targets.sum()
        + 1.0
    )

    return dice.item()


# ============================================================
# Training Function
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_function,
    epoch
):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch} Training"
    )

    for images, masks in progress_bar:

        images = images.to(
            device,
            non_blocking=True
        )

        masks = masks.to(
            device,
            non_blocking=True
        )

        # Add channel dimension to masks
        masks = masks.unsqueeze(1)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = loss_function(
            predictions,
            masks
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    average_loss = (
        total_loss / len(dataloader)
    )

    return average_loss


# ============================================================
# Validation Function
# ============================================================

def validate(
    model,
    dataloader,
    loss_function
):

    model.eval()

    total_loss = 0.0

    total_dice = 0.0

    with torch.no_grad():

        progress_bar = tqdm(
            dataloader,
            desc="Validation"
        )

        for images, masks in progress_bar:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            masks = masks.unsqueeze(1)

            # Forward pass
            predictions = model(images)

            # Loss
            loss = loss_function(
                predictions,
                masks
            )

            # Dice
            dice = dice_score(
                predictions,
                masks
            )

            total_loss += loss.item()

            total_dice += dice

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice:.4f}"
            )

    average_loss = (
        total_loss / len(dataloader)
    )

    average_dice = (
        total_dice / len(dataloader)
    )

    return average_loss, average_dice


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("        Brain Tumor U-Net Training")
    print("========================================")

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_loader, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=BATCH_SIZE
    )

    print("\nTrain batches:")
    print(len(train_loader))

    print("\nValidation batches:")
    print(len(val_loader))

    # --------------------------------------------------------
    # Create Model
    # --------------------------------------------------------

    print("\nCreating U-Net...")

    model = UNet()

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_function = DiceBCELoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_dice = 0.0

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):

        print("\n")
        print("========================================")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print("========================================")

        # Training
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            epoch
        )

        # Validation
        val_loss, val_dice = validate(
            model,
            val_loader,
            loss_function
        )

        # Results
        print("\nEpoch Results:")

        print(
            f"Training Loss:   {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Dice: {val_dice:.4f}"
        )

        # ----------------------------------------------------
        # Save Best Model
        # ----------------------------------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            torch.save(
                model.state_dict(),
                MODEL_SAVE_PATH
            )

            print(
                f"\nBest model saved!"
            )

            print(
                f"Best Dice: {best_dice:.4f}"
            )

    print("\n========================================")
    print("Training Finished")
    print("========================================")

    print(
        f"\nBest Validation Dice: {best_dice:.4f}"
    )

    print(
        f"Model saved as: {MODEL_SAVE_PATH}"
    )