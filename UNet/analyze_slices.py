from pathlib import Path
import nibabel as nib
import random


# =========================================================
# SETTINGS
# =========================================================

DATASET_PATH = Path(
    r"E:\3.2\Brain Tumor Research\BrainTumor_UNet\Datasets\BraTS2018"
)

TRAINING_PATH = (
    DATASET_PATH
    / "MICCAI_BraTS_2018_Data_Training"
)

TRAIN_RATIO = 0.80

RANDOM_SEED = 42


# =========================================================
# FIND ALL PATIENTS
# =========================================================

def get_all_patients():

    patients = []

    for tumor_type in ["HGG", "LGG"]:

        tumor_path = TRAINING_PATH / tumor_type

        for patient_path in sorted(tumor_path.iterdir()):

            if not patient_path.is_dir():
                continue

            patients.append(patient_path)

    return patients


# =========================================================
# CREATE SAME PATIENT SPLIT
# =========================================================

def create_patient_split():

    patients = get_all_patients()

    random.seed(RANDOM_SEED)

    random.shuffle(patients)

    train_size = int(
        len(patients) * TRAIN_RATIO
    )

    train_patients = patients[:train_size]

    val_patients = patients[train_size:]

    return train_patients, val_patients


# =========================================================
# ANALYZE SLICES
# =========================================================

def analyze_patients(patients, split_name):

    total_slices = 0

    tumor_slices = 0

    empty_slices = 0

    tumor_pixels = 0

    total_pixels = 0


    print(
        f"\nAnalyzing {split_name} patients..."
    )


    # -----------------------------------------------------
    # Process each patient
    # -----------------------------------------------------

    for patient_number, patient_path in enumerate(
        patients,
        start=1
    ):

        patient_id = patient_path.name

        seg_path = (
            patient_path
            / f"{patient_id}_seg.nii"
        )

        seg = nib.load(seg_path)

        # Read segmentation volume
        seg_data = seg.get_fdata()

        num_slices = seg_data.shape[2]


        # -------------------------------------------------
        # Analyze every slice
        # -------------------------------------------------

        for slice_idx in range(num_slices):

            mask = seg_data[:, :, slice_idx]

            total_slices += 1

            total_pixels += mask.size

            # ---------------------------------------------
            # Any tumor pixel?
            # ---------------------------------------------

            if (mask > 0).any():

                tumor_slices += 1

                tumor_pixels += (
                    mask > 0
                ).sum()

            else:

                empty_slices += 1


        # -------------------------------------------------
        # Progress
        # -------------------------------------------------

        if patient_number % 25 == 0:

            print(
                f"Processed "
                f"{patient_number}/"
                f"{len(patients)} patients..."
            )


    # =====================================================
    # CALCULATE STATISTICS
    # =====================================================

    tumor_percentage = (
        tumor_slices
        / total_slices
        * 100
    )

    empty_percentage = (
        empty_slices
        / total_slices
        * 100
    )

    tumor_pixel_percentage = (
        tumor_pixels
        / total_pixels
        * 100
    )


    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    print("\n========================================")

    print(split_name.upper())

    print("========================================")

    print(
        "Total slices:",
        total_slices
    )

    print(
        "Tumor-containing slices:",
        tumor_slices
    )

    print(
        "Empty slices:",
        empty_slices
    )

    print(
        "Tumor slices percentage:",
        f"{tumor_percentage:.2f}%"
    )

    print(
        "Empty slices percentage:",
        f"{empty_percentage:.2f}%"
    )

    print(
        "Tumor pixel percentage:",
        f"{tumor_pixel_percentage:.4f}%"
    )

    print("========================================")


    return {
        "total_slices": total_slices,
        "tumor_slices": tumor_slices,
        "empty_slices": empty_slices,
        "tumor_percentage": tumor_percentage,
        "empty_percentage": empty_percentage,
        "tumor_pixel_percentage": tumor_pixel_percentage
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\nCreating patient-level split...")

    train_patients, val_patients = (
        create_patient_split()
    )


    print("\nTraining patients:")
    print(len(train_patients))

    print("\nValidation patients:")
    print(len(val_patients))


    # -----------------------------------------------------
    # Analyze training set
    # -----------------------------------------------------

    train_stats = analyze_patients(
        train_patients,
        "Training"
    )


    # -----------------------------------------------------
    # Analyze validation set
    # -----------------------------------------------------

    val_stats = analyze_patients(
        val_patients,
        "Validation"
    )


    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n\n")
    print("########################################")
    print("# FINAL SUMMARY")
    print("########################################")

    print("\nTRAINING")

    print(
        "Total:",
        train_stats["total_slices"]
    )

    print(
        "Tumor:",
        train_stats["tumor_slices"]
    )

    print(
        "Empty:",
        train_stats["empty_slices"]
    )

    print(
        "Tumor %:",
        f"{train_stats['tumor_percentage']:.2f}%"
    )


    print("\nVALIDATION")

    print(
        "Total:",
        val_stats["total_slices"]
    )

    print(
        "Tumor:",
        val_stats["tumor_slices"]
    )

    print(
        "Empty:",
        val_stats["empty_slices"]
    )

    print(
        "Tumor %:",
        f"{val_stats['tumor_percentage']:.2f}%"
    )


    print("\n########################################")
    print("Analysis complete.")
    print("########################################")