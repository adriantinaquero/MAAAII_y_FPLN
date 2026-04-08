from torch import nn
from torchvision import models


def create_vgg(device, pretrained=True):
    if pretrained:
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    else:
        vgg = models.vgg16(weights=None)

    vggf = vgg.features

    # congelamos capas si es preentrenado
    if pretrained:
        for param in vggf.parameters():
            param.requires_grad = False

    mymodel = nn.Sequential(
        vggf,
        nn.AdaptiveAvgPool2d(output_size=1),
        nn.Flatten(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1),
        nn.Sigmoid()
    )

    return mymodel.to(device)