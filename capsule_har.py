 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torchvision
from pathlib import Path
import sys,os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))   
from model.cnn_model import InceptionI3d,Unit3D

class LinearHead(nn.Module):
    def __init__(self, width = 28, roi_spatial=7, num_classes=60, dropout=0., bias=False):
        super(LinearHead, self).__init__()        
        self.roi_spatial = roi_spatial
        self.roi_maxpool = nn.AdaptiveAvgPool2d(roi_spatial-2)
        self.width = width
        if dropout > 0:
            self.dp = nn.Dropout(dropout)
        else:
            self.dp = None

    def forward(self, features,roi_f):

        roi_features = []
        if roi_f.shape[0] == 1:
            features = features.squeeze(2)
        else:
            features = features.squeeze()
        w_sf = self.width/1920
        h_sf = self.width/1080

        rois_1 =  torch.zeros([ roi_f.shape[0],5])
        rois_2 = torch.zeros([roi_f.shape[0],5])
 
        for i in range(roi_f.shape[0]):
            # if 
            rois_1[i][0] = i
            rois_1[i][1] = roi_f[i][0] * w_sf
            rois_1[i][2] = roi_f[i][1] * h_sf
            rois_1[i][3] = roi_f[i][2] * w_sf
            rois_1[i][4] = roi_f[i][3] * h_sf
            rois_2[i][0] = i
            rois_2[i][1] = roi_f[i][4] * w_sf
            rois_2[i][2] = roi_f[i][5] * h_sf
            rois_2[i][3] = roi_f[i][6] * w_sf
            rois_2[i][4] = roi_f[i][7] * h_sf

        roi_feats_1 = torchvision.ops.roi_align(features, rois_1, (self.roi_spatial, self.roi_spatial))
        roi_feats_2 = torchvision.ops.roi_align(features, rois_2,(self.roi_spatial-2, self.roi_spatial-2))
        roi_feats_1 = self.roi_maxpool(roi_feats_1)

        return roi_feats_1,roi_feats_2


class Self_Attn(nn.Module):
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1)  
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height)  
        energy =  torch.bmm(proj_query,proj_key)  
        attention = self.softmax(energy)  
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height)  

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention



class CapsNet(nn.Module):

    def __init__(self, pt_path='weights/rgb_charades.pt', P=4, pretrained_load='i3d'):
        super(CapsNet, self).__init__()
        self.P = P
        self.conv1 = InceptionI3d(157, in_channels=3, final_endpoint='Mixed_5c')
        pretrained_weights = torch.load(pt_path)
        weights = self.conv1.state_dict()
        loaded_layers = 0
        for name in weights.keys():
            if name in pretrained_weights.keys():
                weights[name] = pretrained_weights[name]
                loaded_layers += 1

        self.conv1.load_state_dict(weights) 
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.c = 512
        self.r = 8
        self.excitation = nn.Sequential(
            nn.Linear(self.c, self.c // self.r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c // self.r, self.c, bias=False),
            nn.Sigmoid())
        self.excitation0 = nn.Sequential(
            nn.Linear(self.c, self.c // self.r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.c // self.r, self.c, bias=False),
            nn.Sigmoid())
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout3d = nn.Dropout3d(0.1)
        self.dropout2d = nn.Dropout2d(0.2)
        self.softmax = nn.Softmax(dim=1)
   
        self.FC1 = nn.Linear(832,512)
        self.FC2 = nn.Linear(1024,512)
        self.FC3 = nn.Linear(512,17)
        self.FC4 = nn.Linear(1024,17)
        self.FCf =  nn.Linear(512,256)
        self.FCi =   nn.Linear(512,256)
        self.roi_al = LinearHead(width = 28,roi_spatial=9)
        self.roi_fl = LinearHead(width = 14,roi_spatial=7)
        self.convin = nn.Conv2d(832, 512, kernel_size=(1, 1))
        self.convfn = nn.Conv2d(832, 512, kernel_size=(1, 1))
        self.convpt = nn.Conv2d(1024, 512, kernel_size=(1, 1))
        self.convf = nn.Conv2d(832, 512, kernel_size=(1, 1))
 
        self.bn1d = nn.BatchNorm2d(512, eps=0.001, momentum=0.01)
        self.bn2d = nn.BatchNorm2d(512, eps=0.001, momentum=0.01)
        self.bnfd = nn.BatchNorm2d(512, eps=0.001, momentum=0.01)
        self.bnpt = nn.BatchNorm2d(512, eps=0.001, momentum=0.01)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.logits = Unit3D(in_channels=384+384+128+128, output_channels=17,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')

        self.attn1 = Self_Attn( 512, 'relu')
        self.attn2 = Self_Attn( 512,  'relu')

 
    def forward(self, img,roi_f):
        x_i,x_f = self.conv1(img)
        b,_,d,h,w = x_i.shape
        x_i = (x_i).squeeze(-3)
        x_i = self.dropout3d(x_i)
        x,xo = self.roi_al(x_i,roi_f)    
        x_c =  self.relu(self.bn1d(self.convin(x)))
        x_cat,_ = self.attn1(x_c)
        xo_c = self.relu(self.bn2d(self.convfn(xo)))
        xo_cat,_ = self.attn2(xo_c)
        part = torch.cat((x_cat,xo_cat),1)
        part = self.relu(self.bnpt(self.convpt(part)))
        bs, c, _, _ = part.shape
        y0 = self.squeeze(part).view(bs, c)
        y0 = self.excitation0(y0).view(bs, c, 1, 1)
        part = part * y0.expand_as(part)
        main_x = self.relu(self.bnfd(self.convf(x)))
        main_x = nn.AdaptiveAvgPool2d(( 7, 7))(main_x)
        parts_f = main_x + part
        parts_f = self.dropout2d(parts_f)
        parts_f = torch.flatten(nn.AdaptiveAvgPool2d(1)(parts_f),1)
        attn_output = self.FC3(parts_f)
        
        return  attn_output         
    
 

