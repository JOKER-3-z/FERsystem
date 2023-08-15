import os

import numpy 
import cv2
from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import transforms,models


class DAN(nn.Module):
    def __init__(self, num_class=7,num_head=4, pretrained=True,device_id='cpu'):
        super(DAN, self).__init__()
        
        resnet = models.resnet18()
        
        if pretrained:
            checkpoint = torch.load('./weights/resnet18_msceleb.pth',map_location=torch.device(device_id))
            resnet.load_state_dict(checkpoint['state_dict'],strict=True)

        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.num_head = num_head
        for i in range(num_head):
            setattr(self,"cat_head%d" %i, CrossAttentionHead())
        self.sig = nn.Sigmoid()
        self.fc = nn.Linear(512, num_class)
        self.bn = nn.BatchNorm1d(num_class)


    def forward(self, x):
        x = self.features(x)
        heads = []
        for i in range(self.num_head):
            heads.append(getattr(self,"cat_head%d" %i)(x))
        
        heads = torch.stack(heads).permute([1,0,2])
        if heads.size(1)>1:
            heads = F.log_softmax(heads,dim=1)
            
        out = self.fc(heads.sum(dim=1))
        out = self.bn(out)
   
        return out

class CrossAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    def forward(self, x):
        sa = self.sa(x)
        ca = self.ca(sa)

        return ca

class SpatialAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3,padding=1),
            nn.BatchNorm2d(512),
        )
        self.conv_1x3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(1,3),padding=(0,1)),
            nn.BatchNorm2d(512),
        )
        self.conv_3x1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,1),padding=(1,0)),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.conv1x1(x)
        y = self.relu(self.conv_3x3(y) + self.conv_1x3(y) + self.conv_3x1(y))
        y = y.sum(dim=1,keepdim=True) 
        out = x*y
        
        return out 

class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 512),
            nn.Sigmoid()    
        )


    def forward(self, sa):
        sa = self.gap(sa)
        sa = sa.view(sa.size(0),-1)
        y = self.attention(sa)
        out = sa * y
        
        return out
    

class FERModel():
    def __init__(self,parameters):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_transforms = transforms.Compose([
                                    transforms.Resize((parameters['img_height'], parameters['img_wide'])),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=parameters['mean'],
                                    std=parameters['std'])
                                ])
        self.labels = parameters['label']

        self.model = DAN(num_head=parameters['num_head'], num_class=parameters['num_class'], pretrained=False)
        checkpoint = torch.load(parameters['modelpath'],
            map_location=self.device)
        self.model.load_state_dict(checkpoint,strict=False)
        self.model.to(self.device)
        self.model.eval()

    def detect(self,src,boxs):
        for box in boxs:
            x, y, w, h = box.astype(int)
            face=src.crop((x,y, x+w, y+h))
            label='face'
            if face is not None:
                label=self.fer(face)
            img=cv2.cvtColor(numpy.asarray(src),cv2.COLOR_RGB2BGR)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=3)
            cv2.putText(img, label  , (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        thickness=2)
            return img
          

    def fer(self, img):
        img = self.data_transforms(img)
        img = img.view(1,3,224,224)
        img = img.to(self.device)

        with torch.set_grad_enabled(False):
            out= self.model(img)
            print(out)
            _, pred = torch.max(out,1)
            index = int(pred)
            label = self.labels[index]

            return label




