"""Script to compute the FID and KID scores using the cleanfid package on cuda enabled GPU. """

import os

from cleanfid import fid

if __name__ == "__main__":
    dataset_name = "horse2zebra"  # Specify the dataset name you wish to compute the FID and KID scores for.

    PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    DATA_DIR = os.path.join(PROJ_ROOT, "data")
    SAVE_DIR = os.path.join(PROJ_ROOT, "outputs")
    DATASET_PATH = os.path.join(DATA_DIR, dataset_name)
    PROCESSED_DIR = os.path.join(PROJ_ROOT, "processed")
    PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DIR, dataset_name)

    TEST_A_DATA_PATH = os.path.join(DATASET_PATH, "testA")
    TEST_B_DATA_PATH = os.path.join(DATASET_PATH, "testB")

    b_to_a_path = os.path.join(PROCESSED_DATASET_PATH, "b_to_a")
    a_to_b_path = os.path.join(PROCESSED_DATASET_PATH, "a_to_b")

    IMGS_USED_B_TO_A_PATH = os.path.join(b_to_a_path, "images_used")
    FAKE_A_GENERATED_PATH = os.path.join(b_to_a_path, "images_generated")
    RECONSTRUCTED_B_PATH = os.path.join(b_to_a_path, "images_reconstructed")

    IMGS_USED_A_TO_B_PATH = os.path.join(a_to_b_path, "images_used")
    FAKE_B_GENERATED_PATH = os.path.join(a_to_b_path, "images_generated")
    RECONSTRUCTED_A_PATH = os.path.join(a_to_b_path, "images_reconstructed")

    fid_score = fid.compute_fid(TEST_A_DATA_PATH, FAKE_A_GENERATED_PATH, mode="legacy_pytorch")
    kid_score = fid.compute_kid(TEST_B_DATA_PATH, FAKE_A_GENERATED_PATH, mode="legacy_pytorch")

    print(f"The FID score is: {fid_score}  |  The KID score is: {kid_score}.")




