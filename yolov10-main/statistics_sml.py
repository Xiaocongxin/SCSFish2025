import numpy as np
import torch
import math
import os
import cv2


def convert(size, bbox):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x * dw
    y = y * dh
    w = w * dw
    h = h * dh
    return [x, y, w, h]

def img_size(file_name):
    dir_name = file_name.split('_')[0]
    img_name = file_name.split('.')[0]
    img = cv2.imread(os.path.join('/D/images',dir_name, f'{img_name}.jpg'))  # 用于读取图片文件
    h, w = img.shape[:2]
    size = (w,h)
    return size

def cal_ciou(box1, box2, eps=1e-7):
    (x1, y1, w1, h1), (x2, y2, w2, h2) = torch.tensor(box1), torch.tensor(box2)
    w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
    b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
    b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps
 
    iou = inter / union
 
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # smallest enclosing box width
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # smallest enclosing box height
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
    
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
 
    alpha = v / (v - iou + (1 + eps))
    ciou = iou - (rho2 / c2 + v * alpha)
 
    return ciou


def process_files(dir_list,dirs_path, gt_dir, rord):
    count = [0,0,0,0]
    
    # 
    for dir_name in dir_list:
        dir_path = os.path.join(dirs_path, dir_name)
    #
        for file_name in os.listdir(dir_path):
            gttxt_dir = file_name.split('_')[0]
            detr_txt = open(os.path.join(dir_path, file_name), 'r')
            gt_txt = open(os.path.join(gt_dir, gttxt_dir, file_name), 'r')
            size = img_size(file_name)

            bboxes_d = parse_bboxes(detr_txt,size,rord)
            bboxes_g = parse_bboxes(gt_txt,size,1)

            
            matched_d = []
            matched_g = []
            
            while bboxes_d and bboxes_g:
                max_iou = 0
                best_match_d = None
                best_match_g = None
                
                for bbox_d in bboxes_d:
                    for bbox_g in bboxes_g:
                        iou = cal_ciou(bbox_d['bbox'], bbox_g['bbox'])
                        if iou > max_iou:
                            max_iou = iou
                            best_match_d = bbox_d
                            best_match_g = bbox_g
                
                if max_iou > 0.5:
                    matched_d.append(best_match_d)
                    matched_g.append(best_match_g)
                    bboxes_d.remove(best_match_d)
                    bboxes_g.remove(best_match_g)
                    
                    class_name_d = int(best_match_d['class_name'])
                    class_name_g = int(best_match_g['class_name'])
                    if class_name_g == class_name_d:
                        dbox = best_match_g['bbox']
                        # if rord ==0:
                        #     area_ratio = (float(dbox[2]) * float(dbox[3]))/(float(size[0])*float(size[1]))
                        # else:
                        area_ratio = float(dbox[2]) * float(dbox[3])
                        # area_ratio = (float(dbox[2]) * float(dbox[2]))/(float(size[0])*float(size[1]))
                        # class_count[class_name_g] += 1
                        if area_ratio<= 0.0005:
                            # tiny_set.add(image_info['file_name'])
                            count[0] += 1
                        elif area_ratio<= 0.001:
                            # small_set.add(image_info['file_name'])
                            count[1] += 1
                        elif area_ratio<= 0.05:
                            # mid_set.add(image_info['file_name'])
                            count[2] += 1
                        else:
                            # big_set.add(image_info['file_name'])
                            count[3] += 1
                else:
                    break   
        
    return count


def parse_bboxes(file,size,num):
    bboxes = []
    for line in file.readlines(): 
        parts = line.strip().split()  # y=1为yolotxt
        bbox = [float(part) for part in parts[1:5]]  
        class_id = parts[0] 
        if num ==0:
            bbox= convert(size, bbox)
            class_id = str(int(class_id) - 1)
        bboxes.append({'bbox': bbox, 'class_name': class_id})
    return bboxes


yolo_dir = '/D/val/labels'
gt_dir = '/D/labels'
y_path = '/D/detect'
d_path = 'D/output'
yval_list = ['val/labels']
ytest_list = ['val/labels']
dval_list = ['val']
dtest_list= ['test']
class_name = [    
    'acaja','acaja-a','acaja-m','acaja-z',
    'acapy','acapy-a','acapy-m','acapy-z',
    'balun','balun-a','balun-m','balun-z',
    'bodax','bodax-a','bodax-m','bodax-z',
    'cepar','cepar-a','cepar-m','cepar-z',
    'cepur','cepur-a','cepur-m','cepur-z',
    'chaau','chaau-a','chaau-m','chaau-z',
    'chakl','chakl-a','chakl-m','chakl-z',
    'chame','chame-a','chame-m','chame-z',
    'chrma','chrma-a','chrma-m','chrma-z',
    'chrwe','chrwe-a','chrwe-m','chrwe-z',
    'ctecy','ctecy-a','ctecy-m','ctecy-z',
    'ctest','ctest-a','ctest-m','ctest-z',
    'dastr','dastr-a','dastr-m','dastr-z',
    'forfl','forfl-a','forfl-m','forfl-z',
    'halho','halho-a','halho-m','halho-z',
    'hemme','hemme-a','hemme-m','hemme-z',
    'hench','hench-a','hench-m','hench-z',
    'labdi','labdi-a','labdi-m','labdi-z',
    'melvi','melvi-a','melvi-m','melvi-z',
    'nasbr','nasbr-a','nasbr-m','nasbr-z',
    'nasli','nasli-a','nasli-m','nasli-z',
    'parcr','parcr-a','parcr-m','parcr-z',
    'parmu','parmu-a','parmu-m','parmu-z',
    'parpl','parpl-a','parpl-m','parpl-z',
    'scobi','scobi-a','scobi-m','scobi-z',
    'shark','shark-a','shark-m','shark-z',
    'sufbu','sufbu-a','sufbu-m','sufbu-z',
    'zanco','zanco-a','zanco-m','zanco-z',
    'fish','fish-a','fish-m','fish-z',
    ]

fish_species_dict = {
    'acaja':0,
    'acapy':1,
    'balun':2,
    'bodax':3,
    'cepar':4,
    'cepur':5,
    'chaau':6,
    'chakl':7,
    'chame':8,
    'chrma':9,
    'chrwe':10,
    'ctecy':11,
    'ctest':12,
    'dastr':13,
    'forfl':14,
    'halho':15,
    'hemme':16,
    'hench':17,
    'labdi':18,
    'melvi':19,
    'nasbr':20,
    'nasli':21,
    'parcr':22,
    'parmu':23,
    'parpl':24,
    'scobi':25,
    'shark':26,
    'sufbu':27,
    'zanco':28,
    'fish':29,
}
names = [
    'acaja',
    'acapy',
    'balun',
    'bodax',
    'cepar',
    'cepur',
    'chaau',
    'chakl',
    'chame',
    'chrma',
    'chrwe',
    'ctecy',
    'ctest',
    'dastr',
    'forfl',
    'halho',
    'hemme',
    'hench',
    'labdi',
    'melvi',
    'nasbr',
    'nasli',
    'parcr',
    'parmu',
    'parpl',
    'scobi',
    'shark',
    'sufbu',
    'zanco',
    'fish']

count1 = process_files(ytest1_list,y_path, gt_dir, 1)
count2 = process_files(dtest1_list,d_path, gt_dir, 0)
print('county5:  ')
print(count1)
print('countd5:  ')
print(count2)
