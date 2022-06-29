# source https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
import os
import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision

# TODO Download option for multiple datasets.. it could be a function within the class and use if conditionals
# TODO bash script to download datasets?
class ImageDataset(Dataset):
    """Creates a data set to be used for either training or testing."""
    def __init__(self,
                 path: str,
                 kind: str,
                 transform: torchvision.transforms.transforms.Compose = None) -> None:
        """Initialises the data set.

        Args:
            path: Path to the data set.
            kind: Either "train" or "test" to decide which data files to create the data set with.
            transform: Composition of pytorch transforms to transform the data with.
        """
        self.transform = transform
        self.path = path
        self.kind = kind
        self.dataA = glob.glob(os.path.join(self.path, f"{self.kind}A", "*.*"))
        self.dataB = glob.glob(os.path.join(self.path, f"{self.kind}B", "*.*"))
        self.dataA_len = len(self.dataA)
        self.dataB_len = len(self.dataB)
        self.length = min(self.dataA_len, self.dataB_len)

    def __getitem__(self, item):
        """Gets individual items from the data set.

        Args:
            item:

        Returns:

        """
        imageA = self.transform(Image.open(self.dataA[item % self.dataA_len]).convert('RGB'))
        imageB = self.transform(Image.open(self.dataB[item % self.dataB_len]).convert('RGB'))
        return imageA, imageB

    def __len__(self):
        return self.length

if __name__ == "__main__":
    # pass
    pass
    # from torchvision.utils import save_image
    # from torchvision import transforms
    # target_shape = 256
    # tr = transforms.Compose([
    #     transforms.RandomCrop(target_shape),
    #     transforms.ToTensor()
    # ])
    #
    # print("transforms", type(tr))
    # print("--------")
    #
    # PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
    # DATA_DIR = os.path.join(PROJ_ROOT, "data")
    # path = os.path.join(DATA_DIR, "vangogh2photo")
    # print("path ", type(path))
    # print("--------")
    # #
    # #
    # # #
    # dataset = ImageDataset(path=path, kind="test", transform=tr)
    # print(type(dataset))
    # print(dataset[0][0])
    # save_image(dataset[0][0],  f"saved_images/image1.png")
    # save_image(dataset[0][1], f"saved_images/image2.png")
    # lenght_d = dataset.length
    # in_channels = dataset[0][0].size()[0]
    # print(dataset[1][0].size())
    # print(dataset.dataB_len)
    # print(dataset[0][140])
    # print(dataset[4][140])

    # This code is good
