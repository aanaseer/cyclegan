"""Script to initiate evaluations."""

import os
import time

import torch
from torchvision import transforms

import wandb

from train_utils import initialise_model
from datasets import ImageDataset

os.environ["WANDB_MODE"] = "dryrun"


def prepare_test_images_for_evaluation(configuration: dict,
                                       dataset_path: str,
                                       b_to_a_path: str,  # PHOTOS_TO_VANGOGH_PATH
                                       a_to_b_path: str,  # VANGOGH_TO_PHOTOS_PATH
                                       ) -> None:
    """Function used to convert all test images to their fake equivalents and reconstructed versions and save in a
    directory. For instance convert all the images in a test folder A to fake images B (using generatorAB) and those in
    test folder B to fake images A (using generatorBA).

    Args:
        configuration: A dictionary containing,
          "dataset_name" - Name of the dataset (eg: horse2zebra);
          "image_dim" - The size to which images in the dataset should be resized to;
          "image_channels" -The number of channels in the images to be used for training;
          "learning_rate" - The learning rate for the generator and discriminator optimisers;
          "beta1" - Beta 1 value to be used with the Adam optimisation algorithm;
          "beta2" - Beta 2 value to be used with the Adam optimisation algorithm;
          "device" - The device to use, either cpu or cuda;
          "batch_size" - Batch size to be used with the data loader;
          "lambda_cycle" - Hyperparameter for relative importance of cycle loss;
          "lambda_identity" - Hyperparameter for relative importance of identity loss;
          "condition_step" - The number of steps per epoch at which to print logs, save checkpoints and images;
          "save_checkpoint" - Boolean to require whether to save a checkpoint or not;
          "save_images" - Boolean to require whether to save images during the training process;
          "epochs" - Number of epochs to run the training;
          "load_checkpoint" - Boolean to state whether to initialise the models from a checkpoint;
          "checkpoint_file_name" - Name of the checkpoint file from which to initialise the model;
          "data_dir" - The path to the dataset;
          "output_save_dir": The path to the directory to save outputs from the training process;
          "learning_decay": Boolean to state whether to use learning rate decay or not.
        dataset_path: Path to the location of the dataset.
        b_to_a_path: Path to a directory to save the images generated from domain B to domain A.
        a_to_b_path: Path to a directory to save the images generated from domain A to domain B.
    """

    wandbinit = wandb.init(project="evaluations", config=configuration, anonymous="allow")
    config = wandbinit.config
    (generator_AB, generator_BA) = initialise_model(config)[0:2]
    generator_AB.eval()
    generator_BA.eval()

    image_to_tensor_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()])

    dataset = ImageDataset(path=dataset_path, kind="test", transform=image_to_tensor_transformation)

    tensor_to_image_transformation = transforms.ToPILImage()

    for i in range(len(dataset)):
        start = time.time()
        img_A_tensor = dataset[i][0]
        img_B_tensor = dataset[i][1]

        fake_imgA_tensor = generator_BA.forward(img_B_tensor)
        fake_imgB_tensor = generator_AB.forward(img_A_tensor)

        reconstructed_imgB_tensor = generator_AB.forward(fake_imgA_tensor)
        reconstructed_imgA_tensor = generator_BA.forward(fake_imgB_tensor)

        img_A = tensor_to_image_transformation(img_A_tensor)
        img_B = tensor_to_image_transformation(img_B_tensor)

        fake_imgA = tensor_to_image_transformation(fake_imgA_tensor)
        fake_imgB = tensor_to_image_transformation(fake_imgB_tensor)

        reconstructed_imgB = tensor_to_image_transformation(reconstructed_imgB_tensor)
        reconstructed_imgA = tensor_to_image_transformation(reconstructed_imgA_tensor)

        img_B.save(os.path.join(b_to_a_path, "images_used", f"{i}_B_to_create_fake_A.jpg"), "JPEG")
        fake_imgA.save(os.path.join(b_to_a_path, "images_generated", f"{i}_fake_A.jpg"), "JPEG")
        reconstructed_imgB.save(os.path.join(b_to_a_path, "images_reconstructed", f"{i}_reconstructed_B.jpg"), "JPEG")

        img_A.save(os.path.join(a_to_b_path, "images_used", f"{i}_A_to_create_fake_B.jpg"), "JPEG")
        fake_imgB.save(os.path.join(a_to_b_path, "images_generated", f"{i}_fake_B.jpg"), "JPEG")
        reconstructed_imgA.save(os.path.join(a_to_b_path, "images_reconstructed", f"{i}_reconstructed_A.jpg"), "JPEG")

        stop = time.time()
        duration = stop - start
        print(f"Images saved. Iteration {i}/{len(dataset)} completed and took {duration:.3f}s.")


if __name__ == "__main__":

    # Ensure the configuration dictionary below is adjusted according to dataset and checkpoint with which you are
    # initiating evaluation of the model. Further, ensure all the required directories (PROCESSED_DIR) are also in
    # place before running this script.

    configuration = {
        "dataset_name": "horse2zebra",
        "image_dim": 256,
        "image_channels": 3,
        "learning_rate": 0.002,
        "beta1": 0.5,
        "beta2": 0.999,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "load_checkpoint": True,
        "checkpoint_file_name": "Epoch154_horse2zebra_fanciful-totem-127_checkpoint.pth",
        "learning_decay": True
    }

    dataset_name = configuration["dataset_name"]

    PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    DATA_DIR = os.path.join(PROJ_ROOT, "data")
    DATASET_PATH = os.path.join(DATA_DIR, dataset_name)
    PROCESSED_DIR = os.path.join(PROJ_ROOT, "processed")
    PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DIR, dataset_name)

    b_to_a_path = os.path.join(PROCESSED_DATASET_PATH, "b_to_a")
    a_to_b_path = os.path.join(PROCESSED_DATASET_PATH, "a_to_b")

    prepare_test_images_for_evaluation(configuration, DATASET_PATH, b_to_a_path, a_to_b_path)
