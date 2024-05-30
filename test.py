'''
Author: zhouyuchong
Date: 2024-05-29 17:46:10
Description: 
LastEditors: zhouyuchong
LastEditTime: 2024-05-30 14:03:27
'''
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.transforms import Normalize
import cv2
from PIL import Image
from repvgg import create_RepVGG_A0 as create
import numpy as np


def export_onnx(model, im, file_name, opset=12, train=False, dynamic=False, simplify=True):
    # ONNX export
    try:
        import onnx
        f = file_name

        torch.onnx.export(model, im, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['input'],
                          output_names=['output_emotion'],
                          dynamic_axes={'input': {0: 'batch',},  # shape(1,3,640,640)
                                        'output_emotion': {0: 'batch',},  # shape(1,25200,85)
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify
        if simplify:
            try:
                import onnxsim

                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'input': list(im.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print("simplify failed \n{}".format(e))
    except Exception as e:
        print('onnx->trt failure: \n{}'.format(e))

def main():
    # Load model
    model = create(deploy=True)
    # 8 Emotions
    emotions = ("anger","contempt","disgust","fear","happy","neutral","sad","surprise")

    # Initialise model
    device = torch.device("cuda")
    model.to(device)
    model.load_state_dict(torch.load("weights/repvgg.pth"))

    # Save to eval
    cudnn.benchmark = True
    model.eval()
    # x = torch.ones((1, 3, 224, 224)).cuda()
    # export_onnx(model, x, "LPRNet_v2.onnx")

    # print("Done")
    # return



    # frame = cv2.imread('../Emotion-Detection-Pytorch/sad_face_01.png')
    # image = cv2.imread("happy_face_00.png")
    image = cv2.imread("sad_face_00.png")
    # frame = cv2.imread('face2.png')

    # cv_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = image.astype(np.float32)

    # image = cv2.resize(cv_img_rgb, (224, 224))
    # image = np.transpose(image, [2, 0, 1])
    # image = np.expand_dims(image, axis=0)
    # image = torch.tensor(image).float()

    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]
    # normalize_transform = Normalize(mean=mean, std=std)
    # x = normalize_transform(image)

    conf = True
    images = [image]

    with torch.no_grad():
            # Normalise and transform images
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.225, 0.225, 0.225])
            x = torch.stack([transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])(Image.fromarray(image)) for image in images])
            # Feed through the model
            print(x.shape)
            y = model(x.to(device))
            print(y.shape)
            print(y)
            result = []
            for i in range(y.size()[0]):
                # Add emotion to result
                emotion = (max(y[i]) == y[i]).nonzero().item()
                print(emotion)
                # Add appropriate label if required
                result.append([f"{emotions[emotion]}{f' ({100*y[i][emotion].item():.1f}%)' if conf else ''}",emotion])
    print(result)


if __name__ == "__main__":
    main()