import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from model import UNet
from dataloader import get_dataloaders


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = (
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet\Datasets\BraTS2018"
)

MODEL_PATH = "best_model.pth"

OUTPUT_DIR = Path("prediction_visualizations")

THRESHOLD = 0.5

# Number of examples to save
NUM_EXAMPLES = 10


# ============================================================
# DEVICE
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
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("     U-Net Prediction Visualization")
    print("========================================")

    # --------------------------------------------------------
    # Load validation dataset
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    _, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=1
    )

    print(
        f"Validation slices: {len(val_loader)}"
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating U-Net...")

    model = UNet()

    model = model.to(device)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Find tumor-containing examples
    # --------------------------------------------------------

    print("\nSearching for tumor-containing slices...")

    examples_found = 0

    with torch.no_grad():

        for slice_index, (images, masks) in enumerate(
            val_loader
        ):

            # ------------------------------------------------
            # Only select slices containing tumor
            # ------------------------------------------------

            if masks.sum().item() == 0:

                continue

            images_gpu = images.to(
                device,
                non_blocking=True
            )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            outputs = model(
                images_gpu
            )

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # ------------------------------------------------
            # Convert to NumPy
            # ------------------------------------------------

            image = (
                images[0]
                .cpu()
                .numpy()
            )

            target = (
                masks[0]
                .cpu()
                .numpy()
            )

            prediction = (
                predictions[0, 0]
                .cpu()
                .numpy()
            )

            probability = (
                probabilities[0, 0]
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # FLAIR image
            # ------------------------------------------------

            flair = image[0]

            # ------------------------------------------------
            # Create figure
            # ------------------------------------------------

            fig, axes = plt.subplots(
                1,
                5,
                figsize=(20, 4)
            )

            # ------------------------------------------------
            # FLAIR
            # ------------------------------------------------

            axes[0].imshow(
                flair,
                cmap="gray"
            )

            axes[0].set_title(
                "FLAIR"
            )

            axes[0].axis("off")

            # ------------------------------------------------
            # Ground Truth
            # ------------------------------------------------

            axes[1].imshow(
                flair,
                cmap="gray"
            )

            axes[1].imshow(
                target,
                alpha=0.5
            )

            axes[1].set_title(
                "Ground Truth"
            )

            axes[1].axis("off")

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            axes[2].imshow(
                flair,
                cmap="gray"
            )

            axes[2].imshow(
                prediction,
                alpha=0.5
            )

            axes[2].set_title(
                "Prediction"
            )

            axes[2].axis("off")

            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            axes[3].imshow(
                probability
            )

            axes[3].set_title(
                "Prediction Probability"
            )

            axes[3].axis("off")

            # ------------------------------------------------
            # Overlay
            # ------------------------------------------------

            axes[4].imshow(
                flair,
                cmap="gray"
            )

            axes[4].imshow(
                target,
                alpha=0.35
            )

            axes[4].imshow(
                prediction,
                alpha=0.35
            )

            axes[4].set_title(
                "Ground Truth + Prediction"
            )

            axes[4].axis("off")

            # ------------------------------------------------
            # Overall title
            # ------------------------------------------------

            fig.suptitle(
                f"Validation Slice {slice_index}",
                fontsize=14
            )

            plt.tight_layout()

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_path = (
                OUTPUT_DIR
                / f"prediction_{examples_found + 1:02d}_slice_{slice_index}.png"
            )

            plt.savefig(
                output_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close()

            print(
                f"Saved: {output_path}"
            )

            examples_found += 1

            # ------------------------------------------------
            # Stop after required examples
            # ------------------------------------------------

            if examples_found >= NUM_EXAMPLES:

                break

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n========================================")
    print("Visualization Finished")
    print("========================================")

    print(
        f"\nExamples saved: {examples_found}"
    )

    print(
        f"Location: {OUTPUT_DIR.resolve()}"
    )