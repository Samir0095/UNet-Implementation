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

THRESHOLD = 0.5


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
# METRIC CALCULATION
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
    # Recall / Sensitivity
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

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("      Brain Tumor U-Net Evaluation")
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
        f"\nValidation slices: {len(val_loader)}"
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

    # --------------------------------------------------------
    # Metric accumulators
    # --------------------------------------------------------

    all_metrics = []

    tumor_slice_metrics = []

    empty_slice_count = 0
    tumor_slice_count = 0

    correct_empty_predictions = 0

    total_pixels = 0

    total_tumor_pixels = 0
    total_predicted_tumor_pixels = 0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print("\nStarting evaluation...\n")

    with torch.no_grad():

        progress_bar = tqdm(
            val_loader,
            desc="Evaluating"
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

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            outputs = model(images)

            probabilities = torch.sigmoid(
                outputs
            )

            predictions = (
                probabilities >= THRESHOLD
            ).float()

            # ------------------------------------------------
            # Convert to NumPy
            # ------------------------------------------------

            prediction = (
                predictions[0, 0]
                .cpu()
                .numpy()
            )

            target = (
                masks[0, 0]
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # Check whether slice contains tumor
            # ------------------------------------------------

            target_has_tumor = (
                target.sum() > 0
            )

            prediction_has_tumor = (
                prediction.sum() > 0
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                prediction,
                target
            )

            all_metrics.append(
                metrics
            )

            # ------------------------------------------------
            # Separate tumor slices
            # ------------------------------------------------

            if target_has_tumor:

                tumor_slice_count += 1

                tumor_slice_metrics.append(
                    metrics
                )

            else:

                empty_slice_count += 1

                if not prediction_has_tumor:

                    correct_empty_predictions += 1

            # ------------------------------------------------
            # Pixel statistics
            # ------------------------------------------------

            total_pixels += target.size

            total_tumor_pixels += target.sum()

            total_predicted_tumor_pixels += (
                prediction.sum()
            )

            # ------------------------------------------------
            # Progress bar
            # ------------------------------------------------

            progress_bar.set_postfix(
                dice=f"{metrics['dice']:.3f}"
            )

    # ========================================================
    # AGGREGATE RESULTS
    # ========================================================

    def average_metric(
        metrics_list,
        key
    ):

        if len(metrics_list) == 0:

            return 0.0

        return np.mean(
            [
                metric[key]
                for metric in metrics_list
            ]
        )


    # --------------------------------------------------------
    # All slices
    # --------------------------------------------------------

    all_dice = average_metric(
        all_metrics,
        "dice"
    )

    all_iou = average_metric(
        all_metrics,
        "iou"
    )

    all_precision = average_metric(
        all_metrics,
        "precision"
    )

    all_recall = average_metric(
        all_metrics,
        "recall"
    )

    all_specificity = average_metric(
        all_metrics,
        "specificity"
    )

    # --------------------------------------------------------
    # Tumor-containing slices
    # --------------------------------------------------------

    tumor_dice = average_metric(
        tumor_slice_metrics,
        "dice"
    )

    tumor_iou = average_metric(
        tumor_slice_metrics,
        "iou"
    )

    tumor_precision = average_metric(
        tumor_slice_metrics,
        "precision"
    )

    tumor_recall = average_metric(
        tumor_slice_metrics,
        "recall"
    )

    tumor_specificity = average_metric(
        tumor_slice_metrics,
        "specificity"
    )

    # --------------------------------------------------------
    # Empty slice accuracy
    # --------------------------------------------------------

    if empty_slice_count > 0:

        empty_slice_accuracy = (
            correct_empty_predictions
            / empty_slice_count
        )

    else:

        empty_slice_accuracy = 0.0

    # --------------------------------------------------------
    # Tumor pixel percentages
    # --------------------------------------------------------

    actual_tumor_percentage = (
        total_tumor_pixels
        / total_pixels
        * 100
    )

    predicted_tumor_percentage = (
        total_predicted_tumor_pixels
        / total_pixels
        * 100
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n\n========================================")
    print("          EVALUATION RESULTS")
    print("========================================")

    print("\nDataset:")
    print("BraTS2018 Validation Set")

    print("\nTotal validation slices:")
    print(len(all_metrics))

    print("\nTumor-containing slices:")
    print(tumor_slice_count)

    print("\nEmpty slices:")
    print(empty_slice_count)

    print("\n----------------------------------------")
    print("Metrics - ALL SLICES")
    print("----------------------------------------")

    print(
        f"Dice Score:       {all_dice:.4f}"
    )

    print(
        f"IoU / Jaccard:    {all_iou:.4f}"
    )

    print(
        f"Precision:        {all_precision:.4f}"
    )

    print(
        f"Recall:           {all_recall:.4f}"
    )

    print(
        f"Specificity:      {all_specificity:.4f}"
    )

    print("\n----------------------------------------")
    print("Metrics - TUMOR SLICES ONLY")
    print("----------------------------------------")

    print(
        f"Dice Score:       {tumor_dice:.4f}"
    )

    print(
        f"IoU / Jaccard:    {tumor_iou:.4f}"
    )

    print(
        f"Precision:        {tumor_precision:.4f}"
    )

    print(
        f"Recall:           {tumor_recall:.4f}"
    )

    print(
        f"Specificity:      {tumor_specificity:.4f}"
    )

    print("\n----------------------------------------")
    print("Empty Slice Performance")
    print("----------------------------------------")

    print(
        f"Correctly empty:  {correct_empty_predictions}"
    )

    print(
        f"Total empty:      {empty_slice_count}"
    )

    print(
        f"Empty accuracy:   {empty_slice_accuracy:.4f}"
    )

    print("\n----------------------------------------")
    print("Tumor Pixel Statistics")
    print("----------------------------------------")

    print(
        f"Actual tumor pixels:     "
        f"{actual_tumor_percentage:.4f}%"
    )

    print(
        f"Predicted tumor pixels:  "
        f"{predicted_tumor_percentage:.4f}%"
    )

    print("\n========================================")
    print("Evaluation Finished")
    print("========================================")