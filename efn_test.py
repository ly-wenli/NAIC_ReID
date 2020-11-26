from efficientnet_pytorch import EfficientNet
import cv2
import torch
from torch import nn

img = cv2.imread('/home/wenli/MDblureGAN/datasets/GOPRO_Large/train/GOPR0372_07_00/sharp/000048.png')
img = torch.Tensor(img).permute(2,0,1)
img_s = torch.unsqueeze(img,dim=0)

model = EfficientNet.from_pretrained('efficientnet-b7')
model_no = EfficientNet.from_name('efficientnet-b7')

img_s = img_s.cuda()
model = model.cuda()
model_no = model_no.cuda()

model = nn.DataParallel(model)
# model_no = nn.DataParallel(model_no)
no_f = nn.DataParallel(model_no.extract_features)(img_s)
print("$$$$$$$$$$$$$$$$$$")
print(no_f)
print(no_f.shape)
print("$$$$$$$$$$$$$$$$$$")
f = model.extract_features(img_s)
print(f)
print(f.shape)



