# source https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
import os
import glob
from PIL import Image
from torch.utils.data import Dataset

# TODO Download option for multiple datasets.. it could be a function within the class and use if conditionals
# TODO bash script to download datasets?
class ImageDataset(Dataset):
    def __init__(self, path, kind, transform=None):
        # Path input here should be /Users/ammar/OneDrive/Oxford/TT 22/Python-special-topic/CycleGAN/data/horse2zebra
        # kind is either train or test
        self.transform = transform
        self.path = path
        self.kind = kind
        self.dataA = glob.glob(os.path.join(self.path, f"{self.kind}A", "*.*"))
        self.dataB = glob.glob(os.path.join(self.path, f"{self.kind}B", "*.*"))
        self.dataA_len = len(self.dataA)
        self.dataB_len = len(self.dataB)
        self.length = min(self.dataA_len, self.dataB_len)

    def __getitem__(self, item):
        imageA = self.transform(Image.open(self.dataA[item % self.dataA_len]).convert('RGB'))
        imageB = self.transform(Image.open(self.dataB[item % self.dataB_len]).convert('RGB'))
        return imageA, imageB

    def __len__(self):
        return self.length

if __name__ == "__main__":
    pass
    # from torchvision.utils import save_image
    # from torchvision import transforms
    # target_shape = 256
    # tr = transforms.Compose([
    #     transforms.RandomCrop(target_shape),
    #     transforms.ToTensor()
    # ])
    #
    # PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    # DATA_DIR = os.path.join(PROJ_ROOT, "data")
    # path = os.path.join(DATA_DIR, "horse2zebra")
    #
    # dataset = ImageDataset(path=path, kind="test", transform=tr)
    # save_image(dataset[0][0],  f"saved_images/image1.png")
    # save_image(dataset[0][1], f"saved_images/image2.png")
    # lenght_d = dataset.length
    # in_channels = dataset[0][0].size()[0]
    # print(dataset[1][0].size())
    # print(dataset.dataB_len)
    # print(dataset[0][140])
    # print(dataset[4][140])

    # This code is good
