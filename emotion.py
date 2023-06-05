import torch
from torchvision.models import vgg11
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as F

from utils.datasets import letterbox
from PIL import Image

# Load model
model = vgg11()

# 8 Emotions
emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")

def init(device):
    # Initialise model
    global dev
    dev = device
    # Change to classify only 8 features
    model.classifier[-1].out_features = 8
    model.classifier[-1]._parameters["weight"] = model.classifier[-1]._parameters["weight"][:8,:]
    model.classifier[-1]._parameters["bias"] = model.classifier[-1]._parameters["bias"][:8]
    
    # Load weights
    state_dict = torch.load("weights/vgg.pth")
    model.load_state_dict(state_dict)
    

    # Prepare for inference
    cudnn.benchmark = True
    model.to(device)
    model.eval()

def transform(image: Image) -> torch.Tensor:
    target_size = 224
    
    # Resize and crop image
    image = F.resize(image, target_size)
    max_size = max(image.size)
    W, H = image.size
    left_padding = (max_size - W) // 2
    right_padding = max_size - W - left_padding
    top_padding = (max_size - H) // 2
    bottom_padding = max_size - H - top_padding
    image = F.pad(image, padding=(left_padding, top_padding, right_padding, bottom_padding), fill=114)
    image = F.resize(image, target_size)

    # Convert to tensor and normalise
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return image

def detect_emotion(images,conf=True):
    with torch.no_grad():
        # Normalise and transform images
        x = torch.stack([
            transform(
                Image.fromarray(
                    image[:,:,::-1]
                )
            ) for image in images
        ])

        # Feed through the model
        logits = model(x.to(dev))
        probs = torch.softmax(logits, dim=-1)
        result = []
        for prob in probs:
            # Add emotion to result
            idx = prob.argmax().item()
            # Add appropriate label if required
            result.append([f"{emotions[idx]}{f' ({100*prob[idx].item():.1f}%)' if conf else ''}",idx])
    return result