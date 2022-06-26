import os
from discriminator import Discriminator
from generator import Generator
from datasets import ImageDataset
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image

import wandb

PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
DATA_DIR = os.path.join(PROJ_ROOT, "data")

config = {
    "dataset_name": "horse2zebra",
    "image_dim": 256,
    "image_channels": 3,
    "learning_rate": 0.0002,
    "beta1": 0.5,
    "beta2": 0.999,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 1,
    "lambda_cycle": 10,
    "lambda_identity": 0.1,
    "condition_step": 10,
    "save_checkpoint": True,
    "save_images": True,
    "epochs": 150
}

def initialise_model(configurations): # configurations passed here will be wandb.config file
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

    return initialised

def initialise_dataloader(configurations, kind):
    # return the dataloader
    PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    DATA_DIR = os.path.join(PROJ_ROOT, "data")
    dataset_path = os.path.join(DATA_DIR, configurations.dataset_name)

    transform = transforms.Compose([
        transforms.Resize(configurations.image_dim),
        # transforms.RandomCrop(configurations.image_dim),
        transforms.ToTensor()
    ])

    dataset = ImageDataset(path=dataset_path, kind=kind, transform=transform)
    dataloader = DataLoader(dataset, batch_size=configurations.batch_size, shuffle=True)
    return dataloader

load_checkpoint = False
checkpoint_file = "checkpoint.pth"
if load_checkpoint:
    print(f"Loading checkpoint: {checkpoint_file}")
    loaded_checkpoint = torch.load(checkpoint_file, map_location=config["device"])
    generator_AB.load_state_dict(loaded_checkpoint["generator_AB"]),
    generator_BA.load_state_dict(loaded_checkpoint["generator_BA"]),
    discriminator_A.load_state_dict(loaded_checkpoint["discriminator_A"]),
    discriminator_B.load_state_dict(loaded_checkpoint["discriminator_B"]),
    optimiser_generator.load_state_dict(loaded_checkpoint["optimiser_generator"]),
    optimiser_discriminator.load_state_dict(loaded_checkpoint["optimiser_discriminator"])


def compute_discriminator_loss(real_imgA, fake_imgA, discriminatorA, adversarial_loss):
    # eg: real_imageA, fake_imageA, discriminatorA, adversarial loss
    disc_classification_fake_img = discriminatorA(fake_imgA.detach())
    disc_classification_real_img = discriminatorA(real_imgA)
    disc_real_loss = adversarial_loss(disc_classification_real_img, torch.ones_like(disc_classification_real_img))
    disc_fake_loss = adversarial_loss(disc_classification_fake_img, torch.zeros_like(disc_classification_fake_img))
    discriminator_loss = disc_real_loss + disc_fake_loss
    return discriminator_loss

def compute_generator_adversarial_loss(real_imgA, discriminatorB, generatorAB, adversarial_loss):
    # real_imgX, discriminatorY, generatorXY, adversarial loss
    fake_imgB = generatorAB(real_imgA)  # fake_imgB
    disc_classification_fake_img = discriminatorB(fake_imgB)
    adversarial_loss = adversarial_loss(disc_classification_fake_img, torch.ones_like(disc_classification_fake_img))
    return fake_imgB, adversarial_loss

def compute_generator_identity_loss(real_imgA, generatorBA, l1_loss):
    # real_imgX, generatorYX,
    identity_img = generatorBA(real_imgA)
    identity_loss = l1_loss(real_imgA, identity_img)
    return identity_loss

def compute_cycle_consistency_loss(real_imgA, fake_imgB, generatorBA, l1_loss):
    # real_imgX, fake_imgY, generator_YX, l1
    img_cycle = generatorBA(fake_imgB)
    cycle_loss = l1_loss(img_cycle, real_imgA)
    return cycle_loss

def compute_generator_losses_one_path(real_imgA, discriminatorB, generatorAB, generatorBA, adversarial_loss,
                                      l1_loss):
    fake_imgB, generator_adversarial_loss_AB = compute_generator_adversarial_loss(real_imgA, discriminatorB, generatorAB, adversarial_loss)
    identity_loss_AB = compute_generator_identity_loss(real_imgA, generatorBA, l1_loss)
    cycle_consistency_loss_BAB = compute_cycle_consistency_loss(real_imgA, fake_imgB, generatorBA, l1_loss)
    return generator_adversarial_loss_AB, identity_loss_AB, cycle_consistency_loss_BAB


# device = 'cpu'
# lambda_cycle = 1
# lambda_identity = 1
# step = 0
# condition_step = 2
# save_checkpoint = False
# save_images = True


def train(generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator, optimiser_discriminator ,dataloader, configurations,
          adversarial_loss, l1_loss):
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
            # eg: real_imageA, fake_imageA, discriminatorA, adversarial loss
            discriminator_A_loss = compute_discriminator_loss(real_imgA, fake_imgA, discriminator_A, adversarial_loss) # Verify these steps
            discriminator_B_loss = compute_discriminator_loss(real_imgB, fake_imgB, discriminator_B, adversarial_loss) # Verify these steps
            discriminator_loss = (discriminator_A_loss + discriminator_B_loss)/2

            # Discriminator gradient updates
            optimiser_discriminator.zero_grad()
            discriminator_loss.backward()
            optimiser_discriminator.step()

            # Generator training
            # Verify the generator loss functions!!
            generator_adversarial_loss_AB, identity_loss_AB, cycle_consistency_loss_BAB = compute_generator_losses_one_path(
                real_imgA, discriminator_B, generator_AB, generator_BA, adversarial_loss, l1_loss)

            generator_adversarial_loss_BA, identity_loss_BA, cycle_consistency_loss_ABA = compute_generator_losses_one_path(
                real_imgB, discriminator_A, generator_BA, generator_AB, adversarial_loss, l1_loss)

            generator_adversarial_loss = generator_adversarial_loss_AB + generator_adversarial_loss_BA
            generator_identity_loss = identity_loss_AB + identity_loss_BA
            generator_cycle_consistency_loss = cycle_consistency_loss_ABA + cycle_consistency_loss_BAB

            generator_loss = (
                generator_adversarial_loss
                + generator_identity_loss * configurations.lambda_identity
                + generator_cycle_consistency_loss * configurations.lambda_cycle
            )

            optimiser_generator.zero_grad()
            generator_loss.backward()
            optimiser_generator.step()


            if step % configurations.condition_step == 0:
                print(f" Epoch: {epoch}  |  Step: {step}  | Discriminator Loss: {discriminator_loss:.2f}  |  "
                      f"Generator Loss: {generator_loss:.2f}")
                wandb.log({"epoch": epoch, "discriminator_loss": discriminator_loss, "generator_loss": generator_loss}, step=step)

                if configurations.save_images:
                    save_image(fake_imgA * 0.5 + 0.5, f"saved_images/{idx}_imgA_fake.png")
                    save_image(fake_imgB * 0.5 + 0.5, f"saved_images/{idx}_imgB_fake.png")

                if configurations.save_checkpoint:
                    print(f"Saving a checkpoint, Epoch: {epoch}, Step: {step}")
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

# TODO check generator losses DONE
# TODO wrap in WANDB API DONE
# TODO side by side image comparision
# TODO losses and epochs to be printed in console  DONE
# TODO state dicts download and load DONE
# TODO paths for saving the model even without pre-made directory

def pipeline(configurations): # input should be the configurations dictionary
    with wandb.init(project="mmsc-python-project", config=configurations, anonymous="allow"):
      config = wandb.config
      (generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator,
       optimiser_discriminator, adversarial_loss, l1_loss) = initialise_model(config)

      dataloader_train = initialise_dataloader(config, "train")

      # make the model, data, and optimization problem
      # model, train_loader, test_loader, criterion, optimizer = make(config)
      # print(model)
      #
      # # and use them to train the model
      # train(model, train_loader, criterion, optimizer, config)
      train(generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator, optimiser_discriminator,
            dataloader_train, config,
            adversarial_loss, l1_loss)
      # # and test its final performance
      # test(model, test_loader)
      #
    # return model
    return (generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator,
       optimiser_discriminator, adversarial_loss, l1_loss, dataloader_train)

# epochs = 10

(generator_AB, generator_BA, discriminator_A, discriminator_B, optimiser_generator,
       optimiser_discriminator, adversarial_loss, l1_loss, dataloader_train) = pipeline(config)



if __name__ == "__main__":
    # print(dataset[0][0].size()[0])
    pipeline(config)