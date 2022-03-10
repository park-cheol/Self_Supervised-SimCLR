import torchvision.transforms as transforms
from data.gaussian_blur import GaussianBlur


class Transforms:
    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomApply([color_jitter], p=0.8),
                                                   transforms.RandomGrayscale(p=0.2),
                                                   GaussianBlur(kernel_size=int(0.1 * size)),
                                                   transforms.ToTensor()]
                                                  )

        self.test_transform = transforms.Compose([transforms.Resize(size=size),
                                                  transforms.ToTensor()]
                                                 )

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

