from torch import nn
from torchvision import models


def create_vgg(device, pretrained=True):
    if pretrained:
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    else:
        vgg = models.vgg16(weights=None)

    # adaptamos la primera capa a 1 canal
    old_layer = vgg.features[0]
    new_layer = nn.Conv2d(3, 64, kernel_size=3, padding=1)

    # if pretrained:
    #     # inicializamos como suma de canales RGB
    #     new_layer.weight.data = old_layer.weight.sum(dim=1, keepdim=True)
    #     new_layer.bias.data = old_layer.bias.data

    vgg.features[0] = new_layer

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
        nn.Linear(1024, 10),
    )

    return mymodel.to(device)