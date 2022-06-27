"""Helper functions to train the machine learning model along with the training loop."""

from __future__ import annotations
import sys
import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

from tqdm import tqdm

import wandb

from datasets import ImageDataset
from discriminator import Discriminator
from generator import Generator


def initialise_model(configurations: wandb.sdk.wandb_config.Config
                     ) -> tuple[Generator, Generator, Discriminator,
                                Discriminator, torch.optim.adam.Adam, torch.optim.adam.Adam,
                                torch.nn.modules.loss.MSELoss, torch.nn.modules.loss.L1Loss]:
    """Initialises the generators, discriminators, optimisers and the loss criteria.

    Args:
        configurations: A wandb Configuration instance. The Configuration instance provided here must have at least the,
          "device" - The device to use, either cpu or cuda
          "image_channels" - The number of channels in the images to be used for training
          "learning_rate" - The learning rate for the generator and discriminator optimisers
          "beta1" - Beta 1 value to be used with the Adam optimisation algorithm
          "beta2" - Beta 2 value to be used with the Adam optimisation algorithm
          "load_checkpoint" - Boolean to state whether to initialise the models from a checkpoint
          "checkpoint_file_name" - Name of the checkpoint file from which to initialise the model

    Returns:
        A tuple containing the initialised machine learning models, optimisers, and the loss criteria as follows,
          generator_AB - Generator for converting image A (from data set A) to image B (from data set B)
          generator_BA - Generator for converting image B (from data set B) to image A (from data set A)
          discriminator_A - Discriminator to determine if an image belongs to the data set A
          discriminator_B - Discriminator to determine if an image belongs to the data set B
          optimiser_generator - Initialised optimiser for the generators using Adam optimisation algorithm
          optimiser_discriminator - Initialised optimiser for the discriminators using Adam optimisation algorithm
          adversarial_loss  - An instance of Mean Squared Loss to be used during training
          l1_loss - An instance of L1 loss to be used during training
    """
    generator_AB = Generator(configurations.image_channels).to(configurations.device)
    generator_BA = Generator(configurations.image_channels).to(configurations.device)
    discriminator_A = Discriminator(configurations.image_channels).to(configurations.device)
    discriminator_B = Discriminator(configurations.image_channels).to(configurations.device)

    adversarial_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()

    optimiser_generator = torch.optim.Adam(
        list(generator_AB.parameters()) + list(generator_BA.parameters()),
        lr=configurations.learning_rate,
        betas=(configurations.beta2, configurations.beta2)
    )

    optimiser_discriminator = torch.optim.Adam(
        list(discriminator_A.parameters()) + list(discriminator_B.parameters()),
        lr=configurations.learning_rate,
        betas=(configurations.beta2, configurations.beta2)
    )

    initialised = (generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator,
                   optimiser_discriminator, adversarial_loss, l1_loss)

    checkpoint_file = configurations.checkpoint_file_name
    try:
        if configurations.load_checkpoint:
            print(f"==> Loading checkpoint: {checkpoint_file}")
            loaded_checkpoint = torch.load(checkpoint_file, map_location=config["device"])
            generator_AB.load_state_dict(loaded_checkpoint["generator_AB"]),
            generator_BA.load_state_dict(loaded_checkpoint["generator_BA"]),
            discriminator_A.load_state_dict(loaded_checkpoint["discriminator_A"]),
            discriminator_B.load_state_dict(loaded_checkpoint["discriminator_B"]),
            optimiser_generator.load_state_dict(loaded_checkpoint["optimiser_generator"]),
            optimiser_discriminator.load_state_dict(loaded_checkpoint["optimiser_discriminator"])
            print("==> Checkpoint successfully loaded.")
    except FileNotFoundError:
        print(f"==> Checkpoint '{checkpoint_file}' does not exist. Re-run with correct configurations.")
        sys.exit(1)
    return initialised


def initialise_dataloader(configurations: wandb.sdk.wandb_config.Config,
                          kind: str) -> torch.utils.data.dataloader.DataLoader:
    """Initialises the dataloader with either the training or testing data.

    Args:
        configurations: A wandb Configuration instance. The Configuration instance provided here must have at least the,
          "data_dir" - The path to the dataset;
          "dataset_name" - Name of the dataset (eg: horse2zebra);
          "image_dim" - The size to which images in the dataset should be resized to;
          "batch_size" - Batch size to be used with the data loader;
        kind: Argument supplied should be either "train" or "test", this determines whether to use train or test data.

    Returns:
        A torch DataLoader object ready to be used when training the machine learning models.
    """
    dataset_path = os.path.join(configurations.data_dir, configurations.dataset_name)

    transform = transforms.Compose([
        transforms.Resize(configurations.image_dim),
        # transforms.RandomCrop(configurations.image_dim),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(path=dataset_path, kind=kind, transform=transform)
    dataloader = DataLoader(dataset, batch_size=configurations.batch_size, shuffle=True)
    return dataloader


def compute_discriminator_loss(real_imgA: torch.Tensor,
                               fake_imgA: torch.Tensor,
                               discriminatorA: Discriminator,
                               adversarial_loss: torch.nn.modules.loss.MSELoss) -> torch.Tensor:
    """Computes the discriminator loss.

    Args:
        real_imgA: A torch tensor of a real image from the data set A.
        fake_imgA: A fake (generated) image such that it looks as if the image is from data set A.
        discriminatorA: Discriminator to determine if an image belongs to the data set A.
        adversarial_loss: An instance of Mean Squared Loss to be used to compute the discriminator loss.

    Returns:
        The computed discriminator loss.
    """
    disc_classification_fake_img = discriminatorA(fake_imgA.detach())
    disc_classification_real_img = discriminatorA(real_imgA)
    disc_real_loss = adversarial_loss(disc_classification_real_img, torch.ones_like(disc_classification_real_img))
    disc_fake_loss = adversarial_loss(disc_classification_fake_img, torch.zeros_like(disc_classification_fake_img))
    discriminator_loss = disc_real_loss + disc_fake_loss
    return discriminator_loss


def compute_generator_adversarial_loss(real_imgA: torch.Tensor,
                                       discriminatorB: Discriminator,
                                       generatorAB: Generator,
                                       adversarial_loss: torch.nn.modules.loss.MSELoss
                                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the generators adversarial loss.

    Args:
        real_imgA: A torch tensor of a real image from the data set A.
        discriminatorB: Discriminator to determine if an image belongs to the data set B
        generatorAB: Generator for converting image A (from data set A) to image B (from data set B)
        adversarial_loss: An instance of Mean Squared Loss to be used to compute the generator loss.

    Returns:
        torch Tensor's of the fake image of B generated using generatorAB and the computed generator loss.
    """
    fake_imgB = generatorAB(real_imgA)  # fake_imgB
    disc_classification_fake_img = discriminatorB(fake_imgB)
    adversarial_loss_computed = adversarial_loss(disc_classification_fake_img,
                                                 torch.ones_like(disc_classification_fake_img))
    return fake_imgB, adversarial_loss_computed


def compute_generator_identity_loss(real_imgA: torch.Tensor,
                                    generatorBA: Generator,
                                    l1_loss: torch.nn.modules.loss.L1Loss) -> torch.Tensor:
    """Computes the generators identity loss.

    Args:
        real_imgA: A torch tensor of a real image from the data set A.
        generatorBA: Generator for converting image B (from data set B) to image A (from data set A).
        l1_loss: An instance of L1 loss used to compute the identity loss.

    Returns:
        The computed identity loss for the generator as a torch Tensor.
    """
    identity_img = generatorBA(real_imgA)
    identity_loss = l1_loss(real_imgA, identity_img)
    return identity_loss


def compute_cycle_consistency_loss(real_imgA: torch.Tensor,
                                   fake_imgB: torch.Tensor,
                                   generatorBA: Generator,
                                   l1_loss: torch.nn.modules.loss.L1Loss) -> torch.Tensor:
    """Computes the cycle consistency loss of the generator.

    Args:
        real_imgA: A torch tensor of a real image from the data set A.
        fake_imgB: A torch tensor of a real image from the data set B.
        generatorBA: Generator for converting image B (from data set B) to image A (from data set A)
        l1_loss: An instance of L1 loss used to compute the cycle consistency loss.

    Returns:
        The computed cycle consistency loss for the generator as a torch Tensor.
    """
    img_cycle = generatorBA(fake_imgB)
    cycle_loss = l1_loss(img_cycle, real_imgA)
    return cycle_loss


def compute_generator_losses_one_path(real_imgA: torch.Tensor,
                                      discriminatorB: Discriminator,
                                      generatorAB: Generator,
                                      generatorBA: Generator,
                                      adversarial_loss: torch.nn.modules.loss.MSELoss,
                                      l1_loss: torch.nn.modules.loss.L1Loss
                                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the generator loss for one path.

    Args:
        real_imgA: A torch tensor of a real image from the data set A.
        discriminatorB: Discriminator to determine if an image belongs to the data set B
        generatorAB: Generator for converting image A (from data set A) to image B (from data set B)
        generatorBA: Generator for converting image B (from data set B) to image A (from data set A).
        adversarial_loss: An instance of Mean Squared Loss criterion.
        l1_loss: An instance of L1 loss criterion.

    Returns:
        The computed adversarial, identity, and cycle consistency loss for one path of the generator as a torch Tensor.
    """
    fake_imgB, generator_adversarial_loss_AB = compute_generator_adversarial_loss(real_imgA, discriminatorB,
                                                                                  generatorAB, adversarial_loss)
    identity_loss_AB = compute_generator_identity_loss(real_imgA, generatorBA, l1_loss)
    cycle_consistency_loss_BAB = compute_cycle_consistency_loss(real_imgA, fake_imgB, generatorBA, l1_loss)
    return generator_adversarial_loss_AB, identity_loss_AB, cycle_consistency_loss_BAB


def train(generator_AB: Generator,
          generator_BA: Generator,
          discriminator_A: Discriminator,
          discriminator_B: Discriminator,
          optimiser_generator: torch.optim.adam.Adam,
          optimiser_discriminator: torch.optim.adam.Adam,
          dataloader: torch.utils.data.dataloader.DataLoader,
          configurations: wandb.sdk.wandb_config.Config,
          adversarial_loss: torch.nn.modules.loss.MSELoss,
          l1_loss: torch.nn.modules.loss.L1Loss
          ) -> None:
    """Starts the training loop.

    Args:
        generator_AB: Generator for converting image A (from data set A) to image B (from data set B).
        generator_BA: Generator for converting image B (from data set B) to image A (from data set A).
        discriminator_A: Discriminator to determine if an image belongs to the data set A.
        discriminator_B: Discriminator to determine if an image belongs to the data set B.
        optimiser_generator: Initialised optimiser for the generators using Adam optimisation algorithm.
        optimiser_discriminator: Initialised optimiser for the discriminators using Adam optimisation algorithm.
        dataloader: A torch DataLoader with training data loaded.
        configurations: A wandb Configuration instance. The Configuration instance provided here must have at least the,
          "epochs" - Number of epochs to run the training;
          "device" - The device to use, either cpu or cuda;
          "lambda_identity" - Hyperparameter for relative importance of identity loss;
          "lambda_cycle" - Hyperparameter for relative importance of cycle loss;
          "condition_step" - The number of steps per epoch at which to print logs, save checkpoints and images.
          "save_images" - Boolean to require whether to save images during the training process.
          "save_checkpoint" - Boolean to require whether to save a checkpoint or not.
        adversarial_loss: An instance of Mean Squared Loss criterion.
        l1_loss: An instance of L1 loss criterion.
    """
    wandb.watch(models=(generator_AB, generator_BA, discriminator_A, discriminator_B), log="all", log_freq=1)
    for epoch in range(configurations.epochs):
        step = 0
        for idx, (imgA, imgB) in enumerate(tqdm(dataloader)):
            real_imgA = imgA.to(configurations.device)
            real_imgB = imgB.to(configurations.device)

            # save_image(real_imgA * 0.5 + 0.5, f"saved_images/real_imgA_{idx}.png")
            # save_image(real_imgB * 0.5 + 0.5, f"saved_images/real_imgB_{idx}.png")

            # Discriminator training
            with torch.no_grad():
                fake_imgA = generator_BA(real_imgB)
                fake_imgB = generator_AB(real_imgA)

            discriminator_A_loss = compute_discriminator_loss(real_imgA, fake_imgA, discriminator_A, adversarial_loss)
            discriminator_B_loss = compute_discriminator_loss(real_imgB, fake_imgB, discriminator_B, adversarial_loss)
            discriminator_loss = (discriminator_A_loss + discriminator_B_loss)/2

            # Discriminator gradient updates
            optimiser_discriminator.zero_grad()
            discriminator_loss.backward()
            optimiser_discriminator.step()

            # Generator training
            generator_adversarial_loss_AB, identity_loss_AB, cycle_consistency_loss_BAB = compute_generator_losses_one_path(
                real_imgA, discriminator_B, generator_AB, generator_BA, adversarial_loss, l1_loss)

            generator_adversarial_loss_BA, identity_loss_BA, cycle_consistency_loss_ABA = compute_generator_losses_one_path(
                real_imgB, discriminator_A, generator_BA, generator_AB, adversarial_loss, l1_loss)

            # Generator losses
            generator_adversarial_loss = generator_adversarial_loss_AB + generator_adversarial_loss_BA
            generator_identity_loss = identity_loss_AB + identity_loss_BA
            generator_cycle_consistency_loss = cycle_consistency_loss_ABA + cycle_consistency_loss_BAB
            generator_loss = (
                generator_adversarial_loss
                + generator_identity_loss * configurations.lambda_identity
                + generator_cycle_consistency_loss * configurations.lambda_cycle
            )

            # Generator gradient updates
            optimiser_generator.zero_grad()
            generator_loss.backward()
            optimiser_generator.step()

            if step % configurations.condition_step == 0:
                print(f"\nEpoch: {epoch} | Step: {step} | Discriminator Loss: {discriminator_loss:.2f} | "
                      f"Generator Loss: {generator_loss:.2f}")
                wandb.log({"epoch": epoch, "discriminator_loss": discriminator_loss, "generator_loss": generator_loss}, step=step)

                if configurations.save_images:
                    # ox_path = "/home/naseer/cyclegan-mmsc-special-topic/src/saved_images/"
                    # ox_path = "saved_images/"
                    save_image(fake_imgA * 0.5 + 0.5, f"saved_images/{epoch}_{step}_imgA_fake.png")
                    save_image(fake_imgB * 0.5 + 0.5, f"saved_images/{epoch}_{step}_imgB_fake.png")

                if configurations.save_checkpoint:
                    print(f"==> Saving a checkpoint, Epoch: {epoch}, Step: {step}")
                    checkpoint = {
                        "generator_AB": generator_AB.state_dict(),
                        "generator_BA": generator_BA.state_dict(),
                        "discriminator_A": discriminator_A.state_dict(),
                        "discriminator_B": discriminator_B.state_dict(),
                        "optimiser_generator": optimiser_generator.state_dict(),
                        "optimiser_discriminator": optimiser_discriminator.state_dict()
                    }
                    torch.save(checkpoint, f"checkpoint.pth")

            step += 1


def initiate_pipeline(configurations: dict) -> None:
    """Initiates the machine learning pipeline and starts training.

    Args:
        configurations: A dictionary containing,
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
    """
    with wandb.init(project="mmsc-python-project", config=configurations, anonymous="allow"):
        config = wandb.config
        (generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator,
         optimiser_discriminator, adversarial_loss, l1_loss) = initialise_model(config)

        dataloader_train = initialise_dataloader(config, "train")

        train(generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator, optimiser_discriminator,
              dataloader_train, config, adversarial_loss, l1_loss)

# TODO check generator losses DONE
# TODO wrap in WANDB API DONE
# TODO side by side image comparision
# TODO losses and epochs to be printed in console  DONE
# TODO state dicts download and load DONE
# TODO paths for saving the model even without pre-made directory
# TODO repeated parameters to be resolved using a decorator

if __name__ == "__main__":
    pass
    # print(dataset[0][0].size()[0])
    # pipeline(config)
    # initiate_pipeline(config)
    # config = {
    #     "dataset_name": "horse2zebra",
    #     "image_dim": 256,
    #     "image_channels": 3,
    #     "learning_rate": 0.0002,
    #     "beta1": 0.5,
    #     "beta2": 0.999,
    #     "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     "batch_size": 1,
    #     "lambda_cycle": 10,
    #     "lambda_identity": 0.1,
    #     "condition_step": 20,
    #     "save_checkpoint": True,
    #     "save_images": True,
    #     "epochs": 150,
    #     "load_checkpoint": False,
    #     "checkpoint_file_name": "checkpoint.pth",
    #     "data_dir": DATA_DIR
    # }
    #
    # wandbinit = wandb.init(project="type-test", config=config, anonymous="allow")
    # config = wandbinit.config
    # (generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator,
    #  optimiser_discriminator, adversarial_loss, l1_loss) = initialise_model(config)
    # # print(initialised)
    #
    # dataloader1 = initialise_dataloader(config, "train")
    #
    # real_imgA = torch.rand(3,256,256)
    # fake_imgA = torch.rand(3, 256, 256)
    # fake_imgB = torch.rand(3, 256, 256)
    #
    # disc_loss = compute_discriminator_loss(real_imgA, fake_imgA, discriminator_A, adversarial_loss)
    # print(type(disc_loss))
    # print("--GENERATOR LOSSES-adversarial")
    # t, v = compute_generator_adversarial_loss(real_imgA, discriminator_B, generator_AB, adversarial_loss)
    # print(type(t))
    # print(type(v))
    #
    # print("--GENERATOR LOSSES-identity")
    # gen_iden_loss = compute_generator_identity_loss(real_imgA, generator_BA, l1_loss)
    # print(type(gen_iden_loss))
    #
    # print("--GENERATOR LOSSES-cycle")
    # cycl_loss = compute_cycle_consistency_loss(real_imgA, fake_imgB, generator_BA, l1_loss)
    # print(type(cycl_loss))
    #
    # print("--GENERATOR LOSSES-onepath")
    # l1g, l2g, l3g  = compute_generator_losses_one_path(real_imgA, discriminator_B,
    #                                                    generator_AB, generator_BA, adversarial_loss, l1_loss)
    # print(type(l1g))
    # print(type(l2g))
    # print(type(l3g))
    #
    # train(generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator, optimiser_discriminator,
    #       dataloader1, config, adversarial_loss, l1_loss)
