 
import os
import time
import numpy as np
import cv2
import torch
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import find_peaks
import ast
from numpy import random
import PIL

req_frames =  8
FrameW = 1920
FrameH = 1080

def quantize(val, to_values):
    best_match = None
    best_match_diff = None
    for other_val in to_values:
        diff = abs(other_val - val)
        if best_match is None or diff < best_match_diff:
            best_match = other_val
            best_match_diff = diff
    index = int(np.where(to_values == best_match)[0]) + 1
    return best_match, index
 

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def normalizer_f(clip, mean, std, inplace=False):
    clip = clip.clone()
    dtype = clip.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])

    return clip

def retrive_roi(img_path):
    with  open(os.path.join(img_path.split('/')[0],'HAR_Test_Frames_rois',img_path.split('/')[-1]+'.csv'), "r") as file:
        FileContent = file.readlines() 
    return FileContent
    
def yolo_C(img_path,rois_vid,train):

    conf = 0.0
    rois_all = []
    
    for rois in rois_vid:
        if img_path.split('/')[-1] == rois.split(',')[0].split('/')[1]:
            hu_m = rois.split(',')[1:]
            for i in range(len(hu_m)//6):
                rois_all.append((hu_m[i*6],float(hu_m[i*6+1]),int(hu_m[i*6+2]),int(hu_m[i*6+3]),int(hu_m[i*6+4]),int(hu_m[i*6+5])))
    
    if rois_all is not None:
        iou = 0
        area = 20 
        h_ = None
        roi_status = 0
        human_status = 0
        for n_ in rois_all:
            if str(n_[0][:-1]) == "Human ROI":
                if float(n_[1])>conf:
                    conf = float(n_[1])
                    box_Lathe = [n_[4],n_[2],n_[5],n_[3]]
                    n_1,n_2,n_3,n_4 = n_[2],n_[3],n_[4],n_[5]
                    roi_status = 1
            else:
                human_status = 1

        if human_status and roi_status:

            for hu in rois_all:
                if hu[0][:-1] == "Human": 
                    box1 = [n_[4],n_[2],n_[5],n_[3]]
                    int_cb= bb_intersection_over_union(box1, box_Lathe)
                    if  iou<=int_cb and ((hu[5]-hu[4])+(hu[3]-hu[2]))>area:
                        iou = int_cb
                        
                        area = (hu[5]-hu[4])+(hu[3]-hu[2]) 
                        h_ = hu

            y,yh,x,xw = int(min(h_[2],int(n_1))),int(max(h_[3],int(n_2))), int(min(h_[4],int(n_3))),int(max(h_[5],int(n_4)))
            yp,yph,xp,xpw = int(n_1),int(n_2), int(n_3),int(n_4)
            h,w = yh-y,xw-x
            hp,wp= yph-yp,xpw-xp
            del_h = 0.2
            del_p = 0.2
            i_h = 0.175
            i_p = 0.175
            
            if train:
                return np.array([int(max(0,x - random.uniform(i_h, del_h)*w)),int(max(0,y - random.uniform(i_h, del_h)*h)),int(min(FrameW,xw + random.uniform(i_h, del_h)*w)),int(min(FrameH,yh + random.uniform(i_h, del_h)*h)),\
                    int(max(0,xp - random.uniform(i_p, del_p)*wp)),int(max(0,yp - random.uniform(i_p, del_p)*hp)),int(min(FrameW,xpw + random.uniform(i_p, del_p)*wp)),int(min(FrameH,yph +random.uniform(i_p, del_p)*hp))]) 
            
            return np.array([int(max(0,x-(w*del_h))),int(max(0,y-(h*del_h))),int(min(FrameW,xw+(w*del_h))),int(min(FrameH,yh+(h*del_h))),\
                    int(max(0,xp-(wp*del_p))),int(max(0,yp-(wp*del_p))),int(min(FrameW,xpw+(wp*del_p))),int(min(FrameH,yph+(wp*del_p)))])

        
        elif  roi_status:
            yp,yph,xp,xpw = int(n_1),int(n_2), int(n_3),int(n_4)
            hp,wp= yph-yp,xpw-xp
            del_p = 0.2
            i_p = 0.175
            if train:
                return np.array([ int(max(0,xp - 2*random.uniform(i_p, del_p)*wp)),int(max(0,yp - 2*random.uniform(i_p, del_p)*hp)),int(min(FrameW,xpw + 2*random.uniform(i_p, del_p)*wp)),int(min(FrameH,yph + 2*random.uniform(i_p, del_p)*hp)),\
                     int(max(0,xp - random.uniform(i_p, del_p)*wp)),int(max(0,yp - random.uniform(i_p, del_p)*hp)),int(min(FrameW,xpw + random.uniform(i_p, del_p)*wp)),int(min(FrameH,yph +random.uniform(i_p, del_p)*hp))]) 
            
            return np.array([int(max(0,xp-(2*wp*del_p))),int(max(0,yp-(2*wp*del_p))),int(min(FrameW,xpw+(2*wp*del_p))),int(min(FrameH,yph+(2*wp*del_p))),\
                int(max(0,xp-(wp*del_p))),int(max(0,yp-(wp*del_p))),int(min(FrameW,xpw+(wp*del_p))),int(min(FrameH,yph+(wp*del_p)))])  

        elif  human_status:
            y,yh,x,xw = int(hu_m[2]),int(hu_m[3]), int(hu_m[4]),int(hu_m[5])
            h,w = yh-y,xw-x
            del_h = 0.2
            i_h = 0.175
            if train:
                return np.array([int(max(0,x - random.uniform(i_h, del_h)*w)),int(max(0,y - random.uniform(i_h, del_h)*h)),int(min(FrameW,xw + random.uniform(i_h, del_h)*w)),int(min(FrameH,yh + random.uniform(i_h, del_h)*h)),\
                    int(max(0,x - random.uniform(i_h, del_h)*w)),int(max(0,y - random.uniform(i_h, del_h)*h)),int(min(FrameW,xw + random.uniform(i_h, del_h)*w)),int(min(FrameH,yh + random.uniform(i_h, del_h)*h))]) 

            return np.array([int(max(0,x-(w*del_h))),int(max(0,y-(h*del_h))),int(min(FrameW,xw+(w*del_h))),int(min(FrameH,yh+(h*del_h))),\
                    int(max(0,x-(w*del_h))),int(max(0,y-(h*del_h))),int(min(FrameW,xw+(w*del_h))),int(min(FrameH,yh+(h*del_h)))])
        else:
            return [0,0,FrameW, FrameH , 0,0,FrameW, FrameH]

    else:
        return[0,0,FrameW, FrameH , 0,0,FrameW, FrameH]
        


def get_video_det(video_dir_f,train,frame_idx,peaks):
    al_size = 224
    no_ch = 3     
    video_dir, frame_first, frame_last = video_dir_f.split(".")[0], video_dir_f.split(".")[1],video_dir_f.split(".")[2]
    allign_video_all = []
    roi_f_all = []
    align_video = np.zeros((req_frames,al_size,al_size, no_ch), dtype=np.uint8)
    img_ret, map_ret = [], []    
    rois_vid = retrive_roi(video_dir)
    if frame_first == 'T':
        s_r = frame_idx
        range_p = 1.0 - s_r[0]
        no_div = range_p/req_frames
        f_l  =  s_r[0]
        idx = 0
        frame_label = []
        for pi in range(req_frames):
            f_i = f_l 
            f_l +=  no_div
            r_n = random.uniform(f_i,f_l)
            val,index = quantize(r_n,s_r)
            image_name =  video_dir + '/'+  ('frame_'+'%d.png' % (int(index)))
            img = cv2.imread(image_name)
            img = cv2.resize(img, (al_size,al_size), interpolation = cv2.INTER_AREA)
            align_video[idx] = img  
            idx+=1
        image_n_roi =  video_dir + '/'+  ('frame_'+'%d.png' % (int(index)//2))
        roi_f = yolo_C(image_n_roi,rois_vid,train)
        if len(roi_f) == 2:
            pass      
 
    else:
        image_n_roi =  video_dir + '/'+  ('frame_'+'%d.png' % (int(frame_idx[-1])//2))
        roi_f = yolo_C(image_n_roi,rois_vid,train)
        if len(roi_f) == 2:
            pass 
        skip = (len(frame_idx))//req_frames
        for idx in  range(0,req_frames,1):
            image_name =  video_dir + '/'+  ('frame_'+'%d.png' % (int(frame_idx[idx*skip])))
            img = cv2.imread(image_name)
            img = cv2.resize(img, (al_size,al_size), interpolation = cv2.INTER_AREA)
            align_video[idx] = img  
 
    return np.array(align_video), np.array(roi_f) 

 
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):

        clip, cur_rois,label  = sample['align_video'], sample['roi_f'],sample['label']
        # print(cur_rois)
        if random.random() < self.p:
            if isinstance(clip[0], np.ndarray):
                aug_clip = [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                
                aug_clip = [img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip]


            top, left, bottom, right,top1, left1, bottom1, right1 = cur_rois[3]/FrameH, cur_rois[0]/FrameW, cur_rois[1]/FrameH, cur_rois[2]/FrameW,\
            cur_rois[7]/FrameH, cur_rois[4]/FrameW, cur_rois[5]/FrameH, cur_rois[6]/FrameW
            flipped_left = 1.0 - right
            flipped_right = 1.0 - left
            new_left  = flipped_left  
            new_right = flipped_right  
            flipped_left1 = 1.0 - right1
            flipped_right1 = 1.0 - left1
            new_left1  = flipped_left1  
            new_right1 = flipped_right1  
            augmented_rois = [top, new_left, bottom, new_right,top1, new_left1, bottom1, new_right1]
            augmented_rois = [max(0,int(new_left*FrameW)),max(0,int(bottom*FrameH)),max(0,int(new_right*FrameW)),max(0,int(top*FrameH)),max(0,int(new_left1*FrameW)),max(0,int(bottom1*FrameH)),max(0,int(new_right1*FrameW)),max(0,int(top1*FrameH))]
            return {'align_video': np.array(aug_clip), 'roi_f': np.array(augmented_rois), 'label': label}
        return {'align_video': np.array(clip), 'roi_f': np.array(cur_rois), 'label': label}

 

class ToTensor(object):

    def __call__(self, sample):
        image_x, map_x, label = sample['align_video'], sample['roi_f'],sample['label']
        image_x = image_x[:,:,:,::-1].transpose((3, 0, 1, 2))
        image_x = np.array(image_x)
        label_np = np.array([0], dtype=np.int64)
        label_np[0] = label
        return {'align_video': torch.from_numpy(image_x).float(), 'roi_f': torch.from_numpy(map_x).float(), 'label': torch.from_numpy(label_np).long()}


class Normalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        clip, map_x, label = sample['align_video'], sample['roi_f'],sample['label']
        image_x =  normalizer_f(clip, self.mean, self.std)

        return {'align_video': np.array(image_x), 'roi_f': np.array(map_x), 'label': label}

class Trainer(Dataset):

    def __init__(self, M,train, transform=None):

        self.train = train
        if self.train: 
            self.train_files = get_det_annotations(M) 
        else: 
             self.train_files = get_det_annotations_test(M) 
        self.transform = transform
        

    def __len__(self):
        return len(self.train_files)

    
    def __getitem__(self, idx):
        image_path = self.train_files[idx][0]
        label = str(self.train_files[idx][1])    
        frame_idx = self.train_files[idx][2]
        peaks = self.train_files[idx][3]
        align_video, roi_f = get_video_det(image_path,self.train,frame_idx,peaks)
        sample = {'align_video': align_video, 'roi_f': roi_f, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
