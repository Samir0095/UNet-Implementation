import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class BraTSDataset(Dataset):

    def __init__(self, patient_list):

        self.samples = []

        self.patient_cache = {}

        for patient_path in patient_list:

            patient_path = Path(patient_path)

            patient_id = patient_path.name

            flair_path = (
                patient_path
                / f"{patient_id}_flair.nii"
            )

            flair_img = nib.load(flair_path)

            num_slices = flair_img.shape[2]

            for slice_idx in range(num_slices):

                self.samples.append(
                    (
                        patient_path,
                        patient_id,
                        slice_idx
                    )
                )

        print(
            f"Created dataset with {len(self.samples)} slices."
        )

    def __len__(self):

        return len(self.samples)

    def _load_patient(
        self,
        patient_path,
        patient_id
    ):

        patient_key = str(patient_path)

        if patient_key in self.patient_cache:

            return self.patient_cache[patient_key]

        flair = nib.load(
            patient_path / f"{patient_id}_flair.nii"
        )

        t1 = nib.load(
            patient_path / f"{patient_id}_t1.nii"
        )

        t1ce = nib.load(
            patient_path / f"{patient_id}_t1ce.nii"
        )

        t2 = nib.load(
            patient_path / f"{patient_id}_t2.nii"
        )

        seg = nib.load(
            patient_path / f"{patient_id}_seg.nii"
        )

        self.patient_cache[patient_key] = (
            flair,
            t1,
            t1ce,
            t2,
            seg
        )

        return (
            flair,
            t1,
            t1ce,
            t2,
            seg
        )

    def __getitem__(self, index):

        patient_path, patient_id, slice_idx = (
            self.samples[index]
        )

        (
            flair,
            t1,
            t1ce,
            t2,
            seg
        ) = self._load_patient(
            patient_path,
            patient_id
        )

        # ----------------------------------------------------
        # Load individual 2D slices
        # ----------------------------------------------------

        flair_slice = np.asarray(
            flair.dataobj[:, :, slice_idx],
            dtype=np.float32
        )

        t1_slice = np.asarray(
            t1.dataobj[:, :, slice_idx],
            dtype=np.float32
        )

        t1ce_slice = np.asarray(
            t1ce.dataobj[:, :, slice_idx],
            dtype=np.float32
        )

        t2_slice = np.asarray(
            t2.dataobj[:, :, slice_idx],
            dtype=np.float32
        )

        seg_slice = np.asarray(
            seg.dataobj[:, :, slice_idx],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Normalize each modality
        # ----------------------------------------------------

        modalities = [
            flair_slice,
            t1_slice,
            t1ce_slice,
            t2_slice
        ]

        normalized_modalities = []

        for modality in modalities:

            # Brain/non-zero pixels
            nonzero = modality != 0

            if np.any(nonzero):

                mean = modality[nonzero].mean()

                std = modality[nonzero].std()

                if std > 0:

                    modality = (
                        modality - mean
                    ) / std

                else:

                    modality = modality - mean

                # Keep background at zero
                modality[~nonzero] = 0

            normalized_modalities.append(
                modality.astype(np.float32)
            )

        # ----------------------------------------------------
        # Stack modalities
        # ----------------------------------------------------

        image = np.stack(
            normalized_modalities,
            axis=0
        )

        # ----------------------------------------------------
        # Convert segmentation to binary mask
        #
        # 0 = background
        # >0 = tumor
        # ----------------------------------------------------

        mask = (
            seg_slice > 0
        ).astype(np.float32)

        # ----------------------------------------------------
        # Convert to PyTorch tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image.copy()
        )

        mask = torch.from_numpy(
            mask.copy()
        )

        return image, mask


# ============================================================
# Patient-level train/validation split
# ============================================================

def get_patient_split(dataset_path):

    dataset_path = Path(dataset_path)

    training_path = (
        dataset_path
        / "MICCAI_BraTS_2018_Data_Training"
    )

    all_patients = []

    for tumor_type in ["HGG", "LGG"]:

        tumor_path = (
            training_path
            / tumor_type
        )

        for patient_path in sorted(
            tumor_path.iterdir()
        ):

            if patient_path.is_dir():

                all_patients.append(
                    patient_path
                )

    # Reproduce the same split as split_dataset.py
    random.seed(42)

    random.shuffle(all_patients)

    split_index = int(
        0.8 * len(all_patients)
    )

    train_patients = (
        all_patients[:split_index]
    )

    val_patients = (
        all_patients[split_index:]
    )

    return (
        train_patients,
        val_patients
    )


# ============================================================
# Create DataLoaders
# ============================================================

def get_dataloaders(
    dataset_path,
    batch_size=1
):

    print("\nCreating patient-level split...")

    train_patients, val_patients = (
        get_patient_split(dataset_path)
    )

    print(
        f"Training patients: {len(train_patients)}"
    )

    print(
        f"Validation patients: {len(val_patients)}"
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print("\nCreating training dataset...")

    train_dataset = BraTSDataset(
        train_patients
    )

    print("\nCreating validation dataset...")

    val_dataset = BraTSDataset(
        val_patients
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return (
        train_loader,
        val_loader
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    DATASET_PATH = (
        r"E:\3.2\Brain Tumor Research\BrainTumor_UNet\Datasets\BraTS2018"
    )

    print("\n========================================")
    print("        Testing DataLoaders")
    print("========================================")

    train_loader, val_loader = get_dataloaders(
        DATASET_PATH,
        batch_size=1
    )

    print("\nTraining batches:")
    print(len(train_loader))

    print("\nValidation batches:")
    print(len(val_loader))

    # --------------------------------------------------------
    # Test training batch
    # --------------------------------------------------------

    images, masks = next(
        iter(train_loader)
    )

    print("\nTraining batch:")

    print("Images shape:")
    print(images.shape)

    print("Masks shape:")
    print(masks.shape)

    print("Images dtype:")
    print(images.dtype)

    print("Masks dtype:")
    print(masks.dtype)

    print("Mask unique values:")
    print(torch.unique(masks))

    # --------------------------------------------------------
    # Test validation batch
    # --------------------------------------------------------

    images, masks = next(
        iter(val_loader)
    )

    print("\nValidation batch:")

    print("Images shape:")
    print(images.shape)

    print("Masks shape:")
    print(masks.shape)

    print("Images dtype:")
    print(images.dtype)

    print("Masks dtype:")
    print(masks.dtype)

    print("Mask unique values:")
    print(torch.unique(masks))

    print("\n========================================")
    print("DataLoader test PASSED!")
    print("========================================")