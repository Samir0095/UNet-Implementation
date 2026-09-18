import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class BraTSDataset(Dataset):

    def __init__(self, dataset_path):

        self.dataset_path = Path(dataset_path)

        self.samples = []

        # -------------------------------------------------
        # Training data contains HGG and LGG
        # -------------------------------------------------

        training_path = (
            self.dataset_path
            / "MICCAI_BraTS_2018_Data_Training"
        )

        # -------------------------------------------------
        # Find all patients
        # -------------------------------------------------

        for tumor_type in ["HGG", "LGG"]:

            tumor_path = training_path / tumor_type

            for patient_path in sorted(tumor_path.iterdir()):

                if not patient_path.is_dir():
                    continue

                patient_id = patient_path.name

                # -----------------------------------------
                # Use FLAIR header to find number of slices
                # -----------------------------------------

                flair_path = (
                    patient_path
                    / f"{patient_id}_flair.nii"
                )

                flair_img = nib.load(flair_path)

                num_slices = flair_img.shape[2]

                # -----------------------------------------
                # Create one sample for every slice
                # -----------------------------------------

                for slice_idx in range(num_slices):

                    self.samples.append(
                        (
                            patient_path,
                            patient_id,
                            slice_idx
                        )
                    )

        # -------------------------------------------------
        # Cache loaded patients
        # -------------------------------------------------

        self.patient_cache = {}

        print("Total slices:", len(self.samples))


    def __len__(self):

        return len(self.samples)


    def _load_patient(self, patient_path, patient_id):

        """
        Load the NIfTI objects for one patient.

        The image data is not fully loaded into RAM.
        Only the required slices are read later.
        """

        patient_key = str(patient_path)

        # Return cached patient if already loaded
        if patient_key in self.patient_cache:

            return self.patient_cache[patient_key]

        # -------------------------------------------------
        # Load NIfTI files
        # -------------------------------------------------

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

        # Store in cache
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

        patient_path, patient_id, slice_idx = self.samples[index]

        # -------------------------------------------------
        # Load patient's NIfTI objects
        # -------------------------------------------------

        flair, t1, t1ce, t2, seg = self._load_patient(
            patient_path,
            patient_id
        )

        # -------------------------------------------------
        # Read only the required 2D slice
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Combine MRI modalities
        # -------------------------------------------------

        image = np.stack(
            [
                flair_slice,
                t1_slice,
                t1ce_slice,
                t2_slice
            ],
            axis=0
        )

        # -------------------------------------------------
        # Convert segmentation to binary mask
        # -------------------------------------------------

        # Original BraTS labels:
        # 0 = background
        # 1, 2, 4 = tumor regions
        #
        # Binary mask:
        # 0 = background
        # 1 = tumor

        mask = (
            seg_slice > 0
        ).astype(np.float32)

        # -------------------------------------------------
        # Convert to PyTorch tensors
        # -------------------------------------------------

        image = torch.from_numpy(image)

        mask = torch.from_numpy(mask)

        return image, mask


# =========================================================
# TEST DATASET AND DATALOADER
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Create dataset
    # -----------------------------------------------------

    dataset = BraTSDataset(
        r"E:\3.2\Brain Tumor Research\BrainTumor_UNet\Datasets\BraTS2018"
    )

    print("\nDataset length:")
    print(len(dataset))


    # -----------------------------------------------------
    # Test one sample
    # -----------------------------------------------------

    image, mask = dataset[100]

    print("\nSingle sample:")
    print("Image shape:")
    print(image.shape)

    print("Mask shape:")
    print(mask.shape)

    print("Image dtype:")
    print(image.dtype)

    print("Mask dtype:")
    print(mask.dtype)

    print("Mask unique values:")
    print(torch.unique(mask))


    # -----------------------------------------------------
    # Create DataLoader
    # -----------------------------------------------------

    batch_size = 4

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


    # -----------------------------------------------------
    # Get one batch
    # -----------------------------------------------------

    images, masks = next(iter(dataloader))


    # -----------------------------------------------------
    # Display batch information
    # -----------------------------------------------------

    print("\nFirst batch:")

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