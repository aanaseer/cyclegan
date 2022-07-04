import os

import torch
from torch.optim import lr_scheduler
from train_utils import initialise_model
import wandb
from datasets import ImageDataset
from torchvision import transforms
from PIL import Image

os.environ["WANDB_MODE"] = "dryrun"

dataset_name = "vangogh2photo"
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
DATA_DIR = os.path.join(PROJ_ROOT, "data")
SAVE_DIR = os.path.join(PROJ_ROOT, "outputs")
DATASET_PATH = os.path.join(DATA_DIR, dataset_name)
PROCESSED_DIR = os.path.join(PROJ_ROOT, "processed")
PROCESSED_DATASET_PATH = os.path.join(PROCESSED_DIR, dataset_name)

PHOTOS_TO_VANGOGH_PATH = os.path.join(PROCESSED_DATASET_PATH, "photos_to_vangogh")
VANGOGH_TO_PHOTOS_PATH = os.path.join(PROCESSED_DATASET_PATH, "vangogh_to_photos")

# PROJ_ROOT = "/home/naseer/cyclegan-mmsc-special-topic/"
# DATA_DIR = "/scratch/naseer/"
# SAVE_DIR = "/home/naseer/cyclegan-mmsc-special-topic/outputs/"




if __name__ == "__main__":
    # os.environ["WANDB_API_KEY"] = "a99544438dbed97b892d82bce79dd3e091a9d42c"


    configuration = {
        "dataset_name": "vangogh2photo",
        "image_dim": 256,
        "image_channels": 3,
        "learning_rate": 0.002,
        "beta1": 0.5,
        "beta2": 0.999,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "load_checkpoint": True,
        "checkpoint_file_name": "Epoch199_vangogh2photo_checkpoint.pth",
        "learning_decay": True
    }

    wandbinit = wandb.init(project="evaluations", config=configuration, anonymous="allow")
    config = wandbinit.config
    (generator_AB, generator_BA, discriminator_A, discriminator_B) = initialise_model(config)[0:4]
    generator_AB.eval()
    generator_BA.eval()

    target_shape = 256
    image_to_tensor_transformation = transforms.Compose([
        transforms.RandomCrop(target_shape),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(path=DATASET_PATH, kind="test", transform=image_to_tensor_transformation)

    tensor_to_image_transformation = transforms.ToPILImage()
    import time
    for i in range(len(dataset)):
        start = time.time()
        img_A_tensor = dataset[i][0]  # Index 0 is for Van Gogh paintings
        img_B_tensor = dataset[i][1]  # Index 1 is for photographs

        fake_imgA_tensor = generator_BA.forward(img_B_tensor)  # Converts from photo to Van Gogh
        fake_imgB_tensor = generator_AB.forward(img_A_tensor)  # Converts from Van Gogh to photographs

        reconstructed_imgB_tensor = generator_AB.forward(fake_imgA_tensor)  # Convert fake Van Gogh to photograph
        reconstructed_imgA_tensor = generator_BA.forward(fake_imgB_tensor)  # Convert fake photograph to Van Gogh

        img_A = tensor_to_image_transformation(img_A_tensor)   # Actual photograph
        img_B = tensor_to_image_transformation(img_B_tensor)   # Actual Van Gogh painting

        fake_imgA = tensor_to_image_transformation(fake_imgA_tensor)  # A fake Van Gogh painting (using actual photos)
        fake_imgB = tensor_to_image_transformation(fake_imgB_tensor)  # A fake photograph (using Van Gogh paintings)

        reconstructed_imgB = tensor_to_image_transformation(reconstructed_imgB_tensor) # Photograph reconstructed using a fake Van Gogh
        reconstructed_imgA = tensor_to_image_transformation(reconstructed_imgA_tensor) # Van Gogh painting reconstructed using a fake photograph


        img_B.save(os.path.join(PHOTOS_TO_VANGOGH_PATH, "images_used", f"{i}_photograph_to_create_fake_VanGogh.jpg"), "JPEG")
        fake_imgA.save(os.path.join(PHOTOS_TO_VANGOGH_PATH, "images_generated", f"{i}_fake_VanGogh.jpg"), "JPEG")
        reconstructed_imgB.save(os.path.join(PHOTOS_TO_VANGOGH_PATH, "images_reconstructed", f"{i}_reconstructed_photograph.jpg"), "JPEG")

        img_A.save(os.path.join(VANGOGH_TO_PHOTOS_PATH, "images_used", f"{i}_painting_to_create_fake_photo.jpg"), "JPEG")
        fake_imgB.save(os.path.join(VANGOGH_TO_PHOTOS_PATH, "images_generated", f"{i}_fake_photo.jpg"), "JPEG")
        reconstructed_imgA.save(os.path.join(VANGOGH_TO_PHOTOS_PATH, "images_reconstructed", f"{i}_reconstructed_painting.jpg"), "JPEG")

        stop = time.time()
        duration = stop - start
        print(f"Iteration {i} completed and took {duration:.3f}s")



