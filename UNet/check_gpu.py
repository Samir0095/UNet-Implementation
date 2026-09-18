import torch


print("\n========================================")
print("PYTORCH / GPU CHECK")
print("========================================")


print("\nPyTorch version:")
print(torch.__version__)


print("\nCUDA available:")
print(torch.cuda.is_available())


if torch.cuda.is_available():

    print("\nGPU:")
    print(torch.cuda.get_device_name(0))

    properties = torch.cuda.get_device_properties(0)

    total_memory = (
        properties.total_memory
        / (1024 ** 3)
    )

    print("\nTotal GPU memory:")
    print(
        f"{total_memory:.2f} GB"
    )

    print("\nCUDA version:")
    print(torch.version.cuda)


else:

    print("\nNo CUDA GPU detected.")

    print(
        "\nTraining would run on CPU, "
        "which may be very slow for this U-Net."
    )


print("\n========================================")
print("GPU check complete.")
print("========================================")