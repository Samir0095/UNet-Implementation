import torch
import numpy as np
from tqdm import tqdm

from model import UNet
from dataloader import get_dataloaders


# ============================================================
# SETTINGS
# ============================================================

DATASET_PATH = (
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet\Datasets\BraTS2018"
)

MODEL_PATH = "best_model.pth"

BATCH_SIZE = 1

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]


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
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    prediction,
    target
):

    prediction = prediction.astype(bool)
    target = target.astype(bool)

    true_positive = np.logical_and(
        prediction,
        target
    ).sum()

    true_negative = np.logical_and(
        ~prediction,
        ~target
    ).sum()

    false_positive = np.logical_and(
        prediction,
        ~target
    ).sum()

    false_negative = np.logical_and(
        ~prediction,
        target
    ).sum()

    # --------------------------------------------------------
    # Dice
    # --------------------------------------------------------

    dice_denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2 * true_positive
        ) / dice_denominator

    # --------------------------------------------------------
    # IoU
    # --------------------------------------------------------

    iou_denominator = (
        true_positive
        + false_positive
        + false_negative
    )

    if iou_denominator == 0:

        iou = 1.0

    else:

        iou = (
            true_positive
        ) / iou_denominator

    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    precision_denominator = (
        true_positive
        + false_positive
    )

    if precision_denominator == 0:

        precision = 1.0

    else:

        precision = (
            true_positive
        ) / precision_denominator

    # --------------------------------------------------------
    # Recall
    # --------------------------------------------------------

    recall_denominator = (
        true_positive
        + false_negative
    )

    if recall_denominator == 0:

        recall = 1.0

    else:

        recall = (
            true_positive
        ) / recall_denominator

    # --------------------------------------------------------
    # Specificity
    # --------------------------------------------------------

    specificity_denominator = (
        true_negative
        + false_positive
    )

    if specificity_denominator == 0:

        specificity = 1.0

    else:

        specificity = (
            true_negative
        ) / specificity_denominator

    return (
        dice,
        iou,
        precision,
        recall,
        specificity
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("        Threshold Analysis")
    print("========================================")

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print("\nLoading validation dataset...")

    _, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=BATCH_SIZE
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
    # Load trained weights
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

    # ========================================================
    # STORE PREDICTION PROBABILITIES
    # ========================================================

    print("\nRunning model on validation set...")

    all_probabilities = []
    all_targets = []

    with torch.no_grad():

        progress_bar = tqdm(
            val_loader,
            desc="Generating predictions"
        )

        for images, masks in progress_bar:

            images = images.to(
                device,
                non_blocking=True
            )

            outputs = model(
                images
            )

            probabilities = torch.sigmoid(
                outputs
            )

            probability = (
                probabilities[0, 0]
                .cpu()
                .numpy()
            )

            target = (
                masks[0]
                .cpu()
                .numpy()
            )

            all_probabilities.append(
                probability
            )

            all_targets.append(
                target
            )

    print("\nPrediction generation finished.")

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    all_probabilities = np.array(
        all_probabilities
    )

    all_targets = np.array(
        all_targets
    )

    print("\nPrediction array shape:")
    print(all_probabilities.shape)

    print("\nTarget array shape:")
    print(all_targets.shape)

    # ========================================================
    # IDENTIFY TUMOR SLICES
    # ========================================================

    tumor_slice_indices = []

    for i in range(
        len(all_targets)
    ):

        if all_targets[i].sum() > 0:

            tumor_slice_indices.append(i)

    tumor_slice_indices = np.array(
        tumor_slice_indices
    )

    print(
        "\nTumor-containing slices:"
    )

    print(
        len(tumor_slice_indices)
    )

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    results = []

    print("\n")
    print("========================================")
    print("       THRESHOLD RESULTS")
    print("========================================")

    print(
        "\nThreshold | Dice | IoU | Precision | Recall | Specificity"
    )

    print(
        "----------------------------------------------------------"
    )

    for threshold in THRESHOLDS:

        dice_values = []
        iou_values = []
        precision_values = []
        recall_values = []
        specificity_values = []

        # ----------------------------------------------------
        # Only evaluate tumor-containing slices
        # ----------------------------------------------------

        for index in tumor_slice_indices:

            probability = (
                all_probabilities[index]
            )

            target = (
                all_targets[index]
            )

            prediction = (
                probability >= threshold
            )

            (
                dice,
                iou,
                precision,
                recall,
                specificity
            ) = calculate_metrics(
                prediction,
                target
            )

            dice_values.append(dice)
            iou_values.append(iou)
            precision_values.append(precision)
            recall_values.append(recall)
            specificity_values.append(
                specificity
            )

        # ----------------------------------------------------
        # Average metrics
        # ----------------------------------------------------

        mean_dice = np.mean(
            dice_values
        )

        mean_iou = np.mean(
            iou_values
        )

        mean_precision = np.mean(
            precision_values
        )

        mean_recall = np.mean(
            recall_values
        )

        mean_specificity = np.mean(
            specificity_values
        )

        results.append(
            {
                "threshold": threshold,
                "dice": mean_dice,
                "iou": mean_iou,
                "precision": mean_precision,
                "recall": mean_recall,
                "specificity": mean_specificity
            }
        )

        print(
            f"   {threshold:.2f}    | "
            f"{mean_dice:.4f} | "
            f"{mean_iou:.4f} | "
            f"{mean_precision:.4f} | "
            f"{mean_recall:.4f} | "
            f"{mean_specificity:.4f}"
        )

    # ========================================================
    # FIND BEST DICE THRESHOLD
    # ========================================================

    best_result = max(
        results,
        key=lambda x: x["dice"]
    )

    print("\n")
    print("========================================")
    print("       BEST THRESHOLD")
    print("========================================")

    print(
        f"\nBest threshold: "
        f"{best_result['threshold']:.2f}"
    )

    print(
        f"Dice:          "
        f"{best_result['dice']:.4f}"
    )

    print(
        f"IoU:           "
        f"{best_result['iou']:.4f}"
    )

    print(
        f"Precision:     "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall:        "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"Specificity:   "
        f"{best_result['specificity']:.4f}"
    )

    print("\n========================================")
    print("Threshold Analysis Finished")
    print("========================================")