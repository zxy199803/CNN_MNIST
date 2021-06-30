import torchvision
train = torchvision.datasets.MNIST(root='.\data',
                                       train=False,
                                       transform=None,
                                       target_transform=None,
                                       download=True)




