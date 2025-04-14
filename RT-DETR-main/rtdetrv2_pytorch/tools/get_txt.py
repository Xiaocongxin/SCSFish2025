"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig

def create_batches(img_paths, batch_size):
    for i in range(0, len(img_paths), batch_size):
        yield img_paths[i:i + batch_size]


# 保存类别，bbox，score
def save_one_txt(img_name, labels, boxes, scores, save_dir, thrh = 0.25):
    # for i, im in enumerate(img_name):
    # for i in len(scores):
        scr = scores[0]
        # lab = labels[0][scr > thrh]
        # box = boxes[0][scr > thrh]
        lab = labels[0]
        box = boxes[0]
        for j in range(len(lab)):
            line = f"{lab[j]} {box[j][0]} {box[j][1]} {box[j][2]} {box[j][3]} {scr[j]}"
            # with open(os.path.join(save_dir, f'{im}.txt'), 'a') as f:
            with open(os.path.join(save_dir, f'{img_name}.txt'), 'a') as f:
                f.write(line + '\n')

def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        # checkpoint = torch.load(args.resume, map_location='cpu', weights_only=True) 
        checkpoint = torch.load(args.resume, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=True) 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Model().to(args.device)
    model = Model().to(device)
    img_paths = []
    with open(args.im_file, 'r') as pfile:
        for line in pfile:
            image_path = line.strip()
            img_paths.append(image_path)

    save_dir = args.summary_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # for batch in create_batches(img_paths, batch_size = 32):
    #     imgs_name = []
    #     im_datas = []
    #     orig_sizes = []
    #     for img_path in batch:
    #         im_pil = Image.open(img_path).convert('RGB')

    #         img_name = os.path.splitext(os.path.basename(img_path))[0]
    #         imgs_name.append(img_name)

    #         w, h = im_pil.size
    #         orig_size = torch.tensor([w, h])[None].to(args.device)
    #         orig_sizes.append(orig_size)

    #         transforms = T.Compose([
    #         T.Resize((1920, 1920)),
    #         T.ToTensor(),
    #         ])
    #         im_data = transforms(im_pil)[None].to(args.device)
    #         im_datas.append(im_data)

    #     im_data =torch.cat(im_datas,dim=0)
    #     orig_size=torch.cat(orig_sizes,dim=0)
    #     output = model(im_data, orig_size)
    #     labels, boxes, scores = output
    #     save_one_txt(imgs_name, labels, boxes, scores, save_dir, thrh = 0.6)
        # draw([im_pil], labels, boxes, scores)
    for img_path in img_paths:
        im_pil = Image.open(img_path).convert('RGB')

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # imgs_name.append(img_name)

        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)
        # orig_sizes.append(orig_size)

        transforms = T.Compose([
        T.Resize((1920, 1920)),
        T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(args.device)
        # im_datas.append(im_data)

    # im_data =torch.cat(im_datas,dim=0)
        output = model(im_data, orig_size)
        labels, boxes, scores = output
        save_one_txt(img_name, labels, boxes, scores, save_dir, thrh = 0.25)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/rtdetrv2/rtdetrv2_r18vd_120e_cocoval.yml')
    parser.add_argument('-r', '--resume', type=str, default='runs/ouput_dir/best.pth')
    parser.add_argument('-f', '--im-file', type=str, default='/D/debug.txt')
    # parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('-d', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--summary-dir', type=str, help='tensorboard summry', default='runs/summary_dir_debug')
    args = parser.parse_args()
    main(args)
