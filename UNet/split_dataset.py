from pathlib import Path
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

patients = []

for tumor_type in ["HGG", "LGG"]:

    tumor_path = TRAINING_PATH / tumor_type

    for patient_path in sorted(tumor_path.iterdir()):

        if not patient_path.is_dir():
            continue

        patients.append(
            (
                patient_path.name,
                tumor_type,
                patient_path
            )
        )


# =========================================================
# DISPLAY DATASET INFORMATION
# =========================================================

print("\nTotal patients:")
print(len(patients))

print("\nHGG patients:")
print(
    sum(
        1
        for patient in patients
        if patient[1] == "HGG"
    )
)

print("\nLGG patients:")
print(
    sum(
        1
        for patient in patients
        if patient[1] == "LGG"
    )
)


# =========================================================
# SHUFFLE PATIENTS
# =========================================================

random.seed(RANDOM_SEED)

random.shuffle(patients)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

train_size = int(
    len(patients) * TRAIN_RATIO
)

train_patients = patients[:train_size]

val_patients = patients[train_size:]


# =========================================================
# DISPLAY SPLIT INFORMATION
# =========================================================

print("\n----------------------------------------")

print("Training patients:")
print(len(train_patients))

print("\nValidation patients:")
print(len(val_patients))

print("----------------------------------------")


# =========================================================
# CHECK FOR DATA LEAKAGE
# =========================================================

train_ids = set(
    patient[0]
    for patient in train_patients
)

val_ids = set(
    patient[0]
    for patient in val_patients
)

overlap = train_ids.intersection(val_ids)


print("\nPatient overlap:")

if len(overlap) == 0:

    print("NONE")

    print("\nNo patient leakage detected.")

else:

    print(overlap)

    print("\nWARNING: Patient leakage detected!")


# =========================================================
# DISPLAY SOME PATIENTS
# =========================================================

print("\nFirst 10 training patients:")

for patient in train_patients[:10]:

    print(
        patient[0],
        "(",
        patient[1],
        ")"
    )


print("\nFirst 10 validation patients:")

for patient in val_patients[:10]:

    print(
        patient[0],
        "(",
        patient[1],
        ")"
    )


# =========================================================
# COUNT HGG / LGG IN EACH SPLIT
# =========================================================

train_hgg = sum(
    1
    for patient in train_patients
    if patient[1] == "HGG"
)

train_lgg = sum(
    1
    for patient in train_patients
    if patient[1] == "LGG"
)

val_hgg = sum(
    1
    for patient in val_patients
    if patient[1] == "HGG"
)

val_lgg = sum(
    1
    for patient in val_patients
    if patient[1] == "LGG"
)


print("\n----------------------------------------")

print("Training HGG:", train_hgg)
print("Training LGG:", train_lgg)

print("Validation HGG:", val_hgg)
print("Validation LGG:", val_lgg)

print("----------------------------------------")