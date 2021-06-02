import argparse
import os
from torch.utils.data import DataLoader, random_split
import torch
import visdom
import time
import numpy as np
from metrics_mnms import metrics
from seg.unet import Unet, AttU_Net
from utils.dataset import PairDataset, StyleDataset, PairDataset_aug
from seg.seg_utils import DiceFocalLoss, get_dice_ob, AverageMeter,DiceLoss
from seg.seg_eva import dc
import cv2
from utils.core import wct_style_transfer, norm, InfiniteSamplerWrapper, adjust_learning_rate, minmaxscaler, poolingscaler, adain
from utils.wavelet_model import WaveEncoder, WaveDecoder
import ipdb
import csv
model_save_dir = 'model_checkpoints/adv_seg'
dir_val_img = '/data/hxq/zd_code_data/data/mnms/Val/all_data/image'
dir_val_lab = '/data/hxq/zd_code_data/data/mnms/Val/all_data/label/'
iters = 2

def seg_train(args):
    os.makedirs(model_save_dir, exist_ok=True)

    model = AttU_Net().cuda()
    model.load_state_dict(torch.load('model_checkpoints/unet_attention/model_30.pth'))
 
    content_dataset = PairDataset_aug(args.content_dir, args.label_dir, args.image_size)
    # num_data = len(content_dataset)
    # part_data, _ = random_split(content_dataset, [50, num_data-50])

    content_dataloader = DataLoader(content_dataset, batch_size=args.batch_size, shuffle=True)
    
    style_dataset = StyleDataset(args.style_dir, args.image_size)
    style_dataloader = iter(DataLoader(
        style_dataset, batch_size=args.batch_size, sampler=InfiniteSamplerWrapper(style_dataset), drop_last=True))

    val_dataset = PairDataset(dir_val_img, dir_val_lab, args.image_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(len(content_dataloader), len(val_dataset))
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)

    loss_meter = AverageMeter()
    lv_meter = AverageMeter()
    rv_meter = AverageMeter()
    myo_meter = AverageMeter()
    for epoch in range(args.epochs):
        model.train()
        for step, image in enumerate(content_dataloader):
            global_step = epoch * len(content_dataloader) + step
            adjust_learning_rate(args.lr, optimizer, global_step) 
            norm_img, target  = image['norm_img'].cuda(), image['scale_lab'].cuda()
            style = norm(next(style_dataloader).cuda())
            cf, sf = model.first_layer(norm_img), model.first_layer(style)
            csf = adain(cf, sf)
            output = model.other_layer(csf)
            pred = torch.argmax(torch.softmax(output,1), 1).data.cpu().numpy()
            loss = criterion(torch.softmax(output, 1), target)            
            label = torch.argmax(target, 1).data.cpu().numpy()
            res = metrics(label, pred, voxel_size=None, flag=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Step[{step}] loss:{loss.item()} LV:{res[0]} MYO:{res[1]} RV:{res[2]} ')
           
            if (global_step) % 200 == 0:
                for image in val_dataloader:
                    model.eval()
                    input = image['image'].cuda()
                    mask = image['label'].cuda()
                    with torch.no_grad():
                        output = model(input)
                    
                    pred = torch.argmax(torch.softmax(output, 1), 1).data.cpu().numpy()
                    label = torch.argmax(mask, 1).data.cpu().numpy()
                    res = metrics(label, pred, voxel_size=None, flag=True)
                    lv_meter.update(res[0]), myo_meter.update(res[1]), rv_meter.update(res[2])
                # ipdb.set_trace()
                with open('result/log/log_aug.csv',"a") as csvfile: 
                    writer = csv.writer(csvfile)
                    if global_step == 0:
                        writer.writerow(["step","lv_dice","myo_dice","rv_dice","average"])
                    avg = (lv_meter.avg+rv_meter.avg+myo_meter.avg)/3.0
                    writer.writerow([global_step,lv_meter.avg,myo_meter.avg,rv_meter.avg, avg])

                model.train()
                lv_meter.reset(), myo_meter.reset(), rv_meter.reset()
                torch.save(model.state_dict(),
                        f'{model_save_dir}/model_{global_step}.pth')

            
if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=2 python adv_seg_train.py
    parser = argparse.ArgumentParser('unet_adv')
    parser.add_argument('--content_dir', type=str,
                        default='/data/hxq/zd_code_data/data/mnms/Train/labeled/image')
    parser.add_argument('--label_dir', type=str,
                        default='/data/hxq/zd_code_data/data/mnms/Train/labeled/label')
    parser.add_argument('--style_dir', type=str,
                        default='/data/hxq/zd_code_data/data/mnms/Train/ven_C/image')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-6) 
    args = parser.parse_args()
    print(args)
    start = time.time()
    seg_train(args)
    print(f'Used Time: {time.time() - start} s')
