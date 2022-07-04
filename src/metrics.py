"""Script to compute the FID and KID scores using the cleanfid package."""

import os
from cleanfid import fid

if __name__ == "__main__":
    dataset_name = "vangogh2photo"
    PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    DATA_DIR = os.path.join(PROJ_ROOT, "data")
    SAVE_DIR = os.path.join(PROJ_ROOT, "outputs")
    DATASET_PATH = os.path.join(DATA_DIR, dataset_name)
    PROCESSED_DIR = os.path.join(PROJ_ROOT, "processed")
    PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DIR, dataset_name)

    TEST_VANGOGH_DATA_PATH = os.path.join(DATA_DIR, "testA")
    TEST_PHOTO_DATA_PATH = os.path.join(DATA_DIR, "testB")

    PHOTOS_TO_VANGOGH_PATH = os.path.join(PROCESSED_DATASET_PATH, "photos_to_vangogh")
    VANGOGH_TO_PHOTOS_PATH = os.path.join(PROCESSED_DATASET_PATH, "vangogh_to_photos")

    IMGS_USED_PHOTO_TO_VANGOGH_PATH = os.path.join(PHOTOS_TO_VANGOGH_PATH, "images_used")
    FAKE_VANGOGH_GENERATED_PATH = os.path.join(PHOTOS_TO_VANGOGH_PATH, "images_generated")
    RECONSTRUCTED_PHOTO_PATH = os.path.join(PHOTOS_TO_VANGOGH_PATH, "images_reconstructed")

    IMGS_USED_VANGOGH_TO_PHOTOS_PATH = os.path.join(VANGOGH_TO_PHOTOS_PATH, "images_used")
    FAKE_PHOTO_GENERATED_PATH = os.path.join(VANGOGH_TO_PHOTOS_PATH, "images_generated")
    RECONSTRUCTED_VANGOGH_PATH = os.path.join(VANGOGH_TO_PHOTOS_PATH, "images_reconstructed")

    fid_score = fid.compute_fid(TEST_VANGOGH_DATA_PATH, FAKE_VANGOGH_GENERATED_PATH, mode="legacy_pytorch")
    kid_score = fid.compute_fid(TEST_VANGOGH_DATA_PATH, FAKE_VANGOGH_GENERATED_PATH, mode="legacy_pytorch")

    print(f"The FID score is: {fid_score}  |  The KID score is: {kid_score}.")



