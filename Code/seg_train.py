import argparse
import os
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import ipdb
from seg.unet import Unet, AttU_Net
from seg.unet_resnet import Resnet_Unet
from utils.dataset import *
from seg.seg_utils import DiceFocalLoss, AverageMeter
from seg.seg_eva import dc
import csv
from metrics_mnms import metrics

def seg_train(args):

    model = AttU_Net()
    model = model.cuda()
    # model.load_state_dict(torch.load('model_checkpoints/model_199.pth'))
    # model.train()
    
    os.makedirs('result/img_strack', exist_ok=True)
    os.makedirs(args.model_save_dir, exist_ok=True)

    train_dataset = PairDataset_aug(
        args.train_img_dir, args.train_lab_dir, args.image_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_dataset = PairDataset(
        args.val_img_dir, args.val_lab_dir, args.image_size)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    print(len(train_dataloader))
    criterion = DiceFocalLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.lr_min)

    loss_meter = AverageMeter()
    lv_meter = AverageMeter()
    rv_meter = AverageMeter()
    myo_meter = AverageMeter()
    # 1:lv, 2:myo, 3:rv
    for epoch in range(args.epochs):
        model.train()
        for step, image in enumerate(train_dataloader):
            # norm_img = image['image'].cuda()
            # mask = image['label'].cuda()
            ori_img, scale_img = image['ori_img'].cuda(), image['scale_img'].cuda()
            ori_lab, scale_lab  = image['ori_lab'].cuda(), image['scale_lab'].cuda()

            
            # if step%20 == 0:
            #     cv2.imwrite(f'result/img_strack/{step}_img.png', np.transpose(ori_img[4].cpu().clone().detach().numpy(), (1,2,0))*255.0)
            #     cv2.imwrite(f'result/img_strack/{step}_lab.png', np.argmax(ori_lab[4].cpu().clone().detach().numpy(), 0)*85.0)
            # ipdb.set_trace()
            output = model(torch.cat([ori_img, scale_img], dim=0))
            loss = criterion(output, torch.cat([ori_lab, scale_lab], dim=0))
            pred = torch.argmax(output[args.batch_size:], 1).data.cpu().numpy()
            label = torch.argmax(scale_lab, 1).data.cpu().numpy()
            # ipdb.set_trace()
            res = metrics(label, pred, voxel_size=None, flag=True)
            loss_meter.update(loss.item(), args.batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch[{epoch}] Step[{step}] loss:{loss_meter.avg} LV:{res[0]} MYO:{res[1]} RV:{res[2]} ')
        scheduler.step()
        loss_meter.reset()

        for image in valid_dataloader:
            model.eval()
            input = image['image'].cuda()
            mask = image['label'].cuda()
            output = model(input)
            loss = criterion(output, mask)
            pred = torch.argmax(torch.softmax(output, 1), 1).data.cpu().numpy()
            label = torch.argmax(mask, 1).data.cpu().numpy()
            res = metrics(label, pred, voxel_size=None, flag=True)
            lv_meter.update(res[0]), myo_meter.update(res[1]), rv_meter.update(res[2])

        with open(args.log,"a") as csvfile: 

            writer = csv.writer(csvfile)
            if epoch == 0:
                writer.writerow(["Epoch","lv_dice","myo_dice","rv_dice","average"])
            avg = (lv_meter.avg+rv_meter.avg+myo_meter.avg)/3.0
            writer.writerow([epoch,lv_meter.avg,myo_meter.avg,rv_meter.avg, avg])
        lv_meter.reset(), myo_meter.reset(), rv_meter.reset()
        torch.save(model.state_dict(),
                   f'{args.model_save_dir}/model_{epoch}.pth')


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES=1 python seg_train.py

    parser = argparse.ArgumentParser('unet_seg')
    parser.add_argument('--train_img_dir', type=str,
                        default='/home/kevinoop/lzd/data/mnms/Train/labeled/image')
    parser.add_argument('--train_lab_dir', type=str,
                        default='/home/kevinoop/lzd/data/mnms/Train/labeled/label')
    parser.add_argument('--val_img_dir', type=str,
                        default='/home/kevinoop/lzd/data/mnms/Val/all_data/image')
    parser.add_argument('--val_lab_dir', type=str,
                        default='/home/kevinoop/lzd/data/mnms/Val/all_data/label')
    parser.add_argument('--log', type=str, default='result/log/log_aug_attaunet.csv')
    parser.add_argument('--model_save_dir', type=str,
                        default='./model_checkpoints_attaunet/')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_min', type=float, default=5e-6)
    args = parser.parse_args()

    print(args)
    os.makedirs(args.model_save_dir, exist_ok=True)
    start = time.time()
    seg_train(args)
    print(f'Used Time: {time.time() - start} s')
