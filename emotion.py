import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image
from vgg import create_RepVGG_A0 as create

model = create(deploy=True)
def init(device):
    global dev
    dev = device
    model.to(device)
    checkpoint = torch.load("weights/vgg.pth")
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    ckpt = {k.replace('module.', ''):v for k,v in checkpoint.items()}   # strip the names
    model.load_state_dict(ckpt)
    model.linear.out_features = 8
    model.linear._parameters["weight"] = model.linear._parameters["weight"][:8,:]
    model.linear._parameters["bias"] = model.linear._parameters["bias"][:8]
    cudnn.benchmark = True
    model.eval()
emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")
def detect_emotion(images,conf=True):
    with torch.no_grad():
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        x = torch.stack([transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])(Image.fromarray(image)) for image in images])
        y = model(x.to(dev))
        result = []
        for i in range(y.size()[0]):
            emotion = (max(y[i]) == y[i]).nonzero().item()
            result.append([f"{emotions[emotion]}{f' ({100*y[i][emotion].item():.1f}%)' if conf else ''}",emotion])
    return result



if __name__ == '__main__':
    image = Image.open("person.png").convert("RGB")
    test([image])
