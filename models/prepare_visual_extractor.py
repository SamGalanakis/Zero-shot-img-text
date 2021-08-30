import torchvision
import torch








def prep_visual_encoder():
    encoder = torchvision.models.resnet50(pretrained=True)







    return encoder