import torch

class UNet:
    def __init__(self, model_path: str = 'models/torch_model_files/UNet.pt'):
        self.model = torch.jit.load(model_path)

class DeepLabV3:
    def __init__(self, model_path: str = 'models/torch_model_files/DeepLabV3.pt'):
        self.model = torch.jit.load(model_path)

if __name__ =="__main__":
    unet = UNet()
    deeplabv3 = DeepLabV3()
    print(unet.model)
    print(deeplabv3.model)