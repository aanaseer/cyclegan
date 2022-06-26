# image_dim = config["image_dim"]
# transform = transforms.Compose([
#     transforms.RandomCrop(image_dim),
#     transforms.ToTensor()
# ])
#
# dataset_name = config["dataset_name"]
# path = os.path.join(DATA_DIR, dataset_name)
# dataset = ImageDataset(path=path, kind="train", transform=transform)
# dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

# image_channels = config["image_channels"]
# device = config["device"]
# generator_AB = Generator(image_channels).to(device)
# generator_BA = Generator(image_channels).to(device)
# discriminator_AB = Discriminator(image_channels).to(device)
# discriminator_BA = Discriminator(image_channels).to(device)
#
# adversarial_loss = nn.MSELoss()
# cycle_consistency_loss = nn.L1Loss()
#
# learning_rate = config["learning_rate"]
# beta1 = config["beta1"]
# beta2 = config["beta2"]
#
# optimiser_generator = torch.optim.Adam(
#     list(generator_AB.parameters()) + list(generator_BA.parameters()),
#     lr=learning_rate,
#     betas=(beta1, beta2)
# )
#
# optimiser_discriminator = torch.optim.Adam(
#     list(discriminator_AB.parameters()) + list(discriminator_BA.parameters()),
#     lr=learning_rate,
#     betas=(beta1, beta2)
# )
