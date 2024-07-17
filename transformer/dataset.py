import torch
import numpy as np
import random
import pandas as pd 
import os
import json
from typing import List, Tuple


"""
notation:
0: SVG END
1: MASK
2: EOM
"""


EOS = 0
SOS = 1
MASK = 2
NUM_SPECIAL = 3



class SketchData(torch.utils.data.Dataset):

    """Sketch dataset"""

    def __init__(self, meta_file_path, svg_folder, img_folder, MAX_LEN, bdry_len, tokenizer, require_aug=False):

        super().__init__()
        self.maxlen = MAX_LEN
        self.svg_folder = svg_folder
        self.img_folder = img_folder
        with open(meta_file_path) as f:
            mf = json.load(f)
        self.mf = mf

        self.bdry_len = bdry_len

        self.uids = np.arange(len(mf), dtype=np.int16).tolist()


    def __len__(self):

        return len(self.uids)


    def prepare_batch(self, pixel_v, bdry_v):

        keys = np.ones(len(pixel_v))
        padding = np.zeros(self.maxlen-len(pixel_v)).astype(int)  
        pixel_v_flat = np.concatenate([pixel_v, padding], axis=0)
        pixel_v_mask = 1-np.concatenate([keys, padding]) == 1  

        bdry_keys = np.ones(len(bdry_v))
        bdry_padding = np.zeros(self.bdry_len-len(bdry_v)).astype(int)  
        bdry_v_flat = np.concatenate([bdry_v, bdry_padding], axis=0)
        bdry_v_mask = 1-np.concatenate([bdry_keys, bdry_padding]) == 1   
        
        return pixel_v_flat, pixel_v_mask, bdry_v_flat, bdry_v_mask




    def get_ordered_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

        return sorted(spans, key=lambda x: x[0])
    

    
    def __getitem__(self, index):
        # uid = self.uids[index]

        sample_name = self.mf[index]
        json_file = self.svg_folder + sample_name

        with open(json_file) as f:
            data_sample = json.load(f)
        coor_list = data_sample['tokens']
        coor_list = [item + 256 + NUM_SPECIAL for item in coor_list]

        # Vehicle_9_2_car_091.txt   P003_9_0_car_091.json P034_0_2_car_005.json /Chair/P002_16_0_ABO_chair_34.json
        c, filename = sample_name.split('/')[1], sample_name.split('/')[2]
        img_name = c + filename[4:].replace('json', 'txt')
        # print(json_file, sample_name, self.img_folder +  img_name)
        boundary = np.loadtxt(self.img_folder +  '/' + img_name)
        boundary = np.squeeze(boundary) + 128
        boundary = boundary.flatten().tolist()
        


        rand = torch.rand(1).item()
        if rand < 0.9:
            aug_bdry_list = boundary
        else:
            aug_bdry_list = []


        vec_data_bdry = self.clipping(aug_bdry_list, min=NUM_SPECIAL, max=256+NUM_SPECIAL)
        vec_data = self.clipping(coor_list, min=256+NUM_SPECIAL, max=256+4096+NUM_SPECIAL)

        
        bdry = np.concatenate((vec_data_bdry, np.zeros(1)), axis=0)
        pixs = np.concatenate((vec_data, np.zeros(1)), axis=0)

        pix_seq, mask, bdry_seq, bdry_mask = self.prepare_batch(pixs, bdry)
        pix_seq = torch.from_numpy(pix_seq).to(torch.int32)
        bdry_seq = torch.from_numpy(bdry_seq).to(torch.int32)

        return pix_seq, mask, bdry_seq, bdry_mask
    

    def clipping(self, pixel_list, min=0, max=10):

        sequence = np.array(pixel_list)
        sequence = np.clip(sequence, a_min=min, a_max=max)

        return sequence

