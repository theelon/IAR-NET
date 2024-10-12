import time,os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from collections import OrderedDict
from math import sqrt
import numpy as np
import pdb
import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data 
from utils.dataloader import *
from model.capsule_har import *
from model.cnn_model import InceptionI3d
import matplotlib.pyplot as plt
import csv
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))   
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))   
theta = 3.0

device = torch.device('cpu')
print("======================================================================")
print('===> Device Using',device)
cuda = 'store_true'
pretrained = 'weights/model_epoch_28_0_Fold.pth'
test_dir = 'Input' 
save_dir = 'Results'
file_name = "output.txt"

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def clean_file(file_path):
    with open(file_path, "w") as file:
        file.truncate(0)

HARdict =   {
  "Adjusting_tool_post": 0,
  "Carriage_away_to_lathe_machine": 1,
  "Carriage_near_to_lathe_machine":2,
  "Cutting_workpiece-by_tool":3,
  "Fixing_cutting_tool_on_tool_post":4,
  "Loosening_opening_chuck_jaws":5,
  "Operation_on_Tailstock":6,
  "Switching_OFF_lathe_machine":7,
  "Switching_ON_lathe_machine":8,
  "Tightening_chuck_jaws":9,
  "Loading_workpiece": 10,
  "Measuring_dimensions": 11,
   "Miscellaneous": 12,
  "Opening_tool_post": 13,
  "Tightening_tool_post": 14,
  "Using_Paper_for_finishing": 15,
  "Unloading_workpiece":16
}

def get_key_from_value(dictionary, value_search):
    for key, value in dictionary.items():
        if value == value_search:
            return key
    return None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
  global beta,samples_per_cls,nsamples_per_cls

  
  seed = random.randint(1, 10000)
  print("===> Random Seed: ", seed)
  torch.manual_seed(seed)
  cudnn.benchmark = True

  print('===> Building Model')

  model = CapsNet()
  model = model.to(device)

  print("The model has : '{}' Parameters ".format(count_parameters(model)))

  if pretrained:
      if os.path.isfile(pretrained):
          print('===> Loading Pretrained Model')
          weights = torch.load(pretrained,map_location= device)        
          model.load_state_dict(weights['model'].state_dict())
      else:
          print("=> no model found at '{}'".format(pretrained))
          
  print('===> Loading Dataset')

  transform=transforms.Compose([ToTensor(),Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

  print('===> Testing')
  print("######################################################################")
  test_new(transform,model)
  print("######################################################################")
  print("======================================================================")


def key_frames(frame_indices,source_folder):
  for index in frame_indices:
    source_file = f"{source_folder}/frame_{index}.png"
    if not os.path.exists(f"{save_dir}/selected_frames_{source_folder.split('/')[-1]}"):
      os.mkdir(f"{save_dir}/selected_frames_{source_folder.split('/')[-1]}")
    result_file = f"{save_dir}/selected_frames_{source_folder.split('/')[-1]}/frame_{index}.png"
    shutil.copy(source_file, result_file)

def test_new(transform,model):
    model.eval()
    correct = 0
    test_loss=0.0
    labels_all, predictions_all = [],[]
    validation_annot = []
    # for files in os.listdir(os.path.join(valset_dir)):
    #     file = os.listdir(os.path.join(valset_dir,files))
    files = os.path.join(test_dir,'HAR_Test_Frames')
    for video in os.listdir(files):
        path = os.path.join(files,video)
        frames_total = len(os.listdir(path)) 
        
        frame_idx,plot_frame_idx = [],[]
        if frames_total > 0:
            file_r = os.path.join(test_dir,'HAR_Test_Frames_Encoded',path.split('/')[-1]+'.csv')
            df = pd.read_csv(file_r).to_numpy()
            prev = 0
            l_df = len(df)
            r_v_i = np.diagonal(cosine_similarity(df[0:],df[1:]))
            r_v = [round(r,4) for r in r_v_i]
            d_r_v = np.subtract(1,r_v) + np.finfo(np.float32).eps
            peaks, prop = find_peaks(d_r_v, height=0)
            sort_index = np.argsort(prop['peak_heights'])

            avg_sum = 0
            for i in range(int(len(sort_index))):
                avg_sum+= prop['peak_heights'][sort_index[-i]]
            avg_sum = avg_sum/int(len(sort_index))
            old_d_r_v = d_r_v.copy()
            pk_plot = 0
            dummy,dummy_peaks = [],[]
            for p in range(0,int(len(d_r_v)),1):
                if d_r_v[p]>=theta*avg_sum:
                    dummy.append(d_r_v[p])
                    dummy_peaks.append(p+1)
                    d_r_v[p] =  avg_sum                       
                    pk_plot = 1 
            s_r=[]
            j=0
            for dat in d_r_v:  
                j+= dat
                s_r.append(j)
            s_r = s_r/max(s_r)
            
            ############### plots ##########
            x = np.linspace(1, len(s_r),len(s_r))
            plt.plot(x, s_r)

            s_r_old=[]
            j=0
            for dt in old_d_r_v: 
                j+= dt
                s_r_old.append(j)            
            s_r_old = s_r_old/max(s_r_old)
            plt.plot(x, s_r_old,'-')
            plt.plot(x, old_d_r_v, "x")
            plt.plot(dummy_peaks,dummy, "rx")
            plt.xlabel("Frame #")
            plt.savefig(os.path.join(save_dir,str(path.split('/')[-1]) + "_cumulative.png"))
            plt.clf()

            range_p = 1.0 - s_r[0]
            prev_index = 0
            no_div = range_p/req_frames
            f_l  =  s_r[0]
            for pi in range(req_frames):
                f_i = f_l 
                f_l +=  no_div
                r_n = (f_i+f_l)/2 
                val,index = quantize(r_n,s_r)
                if prev_index == int(index):
                    frame_idx.append(int(index)+1)   
                    prev_index = int(index)+1
                else:
                    frame_idx.append(int(index))  
                    prev_index = int(index)               
          
            new_path = path + '.' + str(0) + '.' + str(l_df)
            labels = HARdict[video]
            key_frames(frame_idx,path)
            align_video, roi_f = get_video_det(new_path,False,frame_idx,None)
            sample = {'align_video': align_video, 'roi_f': roi_f, 'label': labels}
            sample = transform(sample)
            align_video, roi_f = sample['align_video'], sample['roi_f']
            align_video,roi_f = np.expand_dims(align_video, axis=0),np.expand_dims(roi_f, axis=0)
            align_video = torch.from_numpy(align_video.astype(np.float32)).float() 
            roi_f = torch.from_numpy(roi_f.astype(np.float32)).float()
            
            # modified_shape = torch.flip(align_video, dims=(2,)) #flipping experiments 

            pred = model(align_video.to(device),roi_f.to(device))
            pred = pred.softmax(dim = 1)
            predictions = torch.argmax(input=pred, axis=1)[0]
            predictions  = int(predictions.cpu().detach())
            print("Model Prediction:",get_key_from_value(HARdict,predictions))
            print("Actual Label:",video)
            print('------------------------------------------------------------')
            labels_all.append(video)
            predictions_all.append(get_key_from_value(HARdict,predictions))
    txt_path = os.path.join(save_dir,file_name)
    clean_file(txt_path)
    max_length = 30   
    with open(txt_path, "a") as file:
      content = "#" * 70
      file.write(content)

      file.write(f"\n{'*** Labels ***':<{max_length + 2}}{'***  Model Prediction ***':<{max_length + 2}}\n")
      for labels, predictions in zip(labels_all, predictions_all):
          file.write(f"{labels:<{max_length +2}}{predictions:<{max_length + 2}}\n")
      file.write(content)
                 
if __name__ == '__main__':
    main()
