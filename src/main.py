from argparse import ArgumentParser
from utils import set_seed
from model import get_model
from sklearn.model_selection import StratifiedKFold
from glob import glob
import monai
from monai.transforms import LoadNifti, Randomizable, apply_transform
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor, RandAffine
from monai.utils import get_seed
from monai.data import NiftiDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, ToTensor
import time
import numpy as np
import pandas as pd
import cv2
import PIL.Image
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
import random

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import roc_auc_score
import albumentations

def parse_args():
    parser = ArgumentParser()

    # experiment configurations
    parser.add_argument('--name', type=str, default='test',
                        help='name of the experiment')
    parser.add_argument('--image-size', type=float, default=256,
                        help='image size for training and inference')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Debug Mode')
    parser.add_argument('--init-lr', type=float, default=3e-4,
                        help='Initial learning rate')
    parser.add_argument('--output-dim', type=int, default=1)
    parser.add_argument('--bs', type=int, default=48, 
                        help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=48,
                        help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=24,
                        help='Number of workers')
    parser.add_arugment('--pretrained-path', type=str, default=None,
                        help='If you want to resume training from a checkpoint, please specify the path.')
    parser.add_arugment('--seed', type=int, default=42)
    
    # more options
    parser.add_argument('--percentage', type=float, default=1.0,
                        help='the percentage of data needed for training.')
    parser.add_argument('--use-amp', type=bool, default=False,
                        help='pytorch native automatic mixed precision training.')
    # placeholder
    parser.add_argument('--gpus', type=list, default=[0,1,2])
    return parser.parse_args()

class BMDataset3D(torch.utils.data.Dataset, Randomizable):
    def __init__(self, csv, mode, transform=None):

        self.csv = csv.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]
    
    def randomize(self) -> None:
        MAX_SEED = np.iinfo(np.uint32).max + 1
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")    

    def __getitem__(self, index):
        self.randomize()
        row = self.csv.iloc[index]
        image = np.load(row.image_path )
        mask = np.load(row.mask_path )
        img = np.stack([image,  mask]) #  channel(2), z, x, y

        if self.transform is not None:
            if isinstance(self.transform, Randomizable):
                self.transform.set_random_state(seed=self._seed)
            img = apply_transform(self.transform, img)

        if self.mode == 'test':
            return img
        else:
            return img, torch.tensor(row['Failure-binary']).float()
        
def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        
    for (data, target) in bar:
        data, target = data.to(device), target.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(data)
                loss = criterion(logits.squeeze(), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(data)
            loss = criterion(logits.squeeze(), target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-100:]) / min(len(train_loss), 100)
        bar.set_description('loss: %.5f, smth: %.5f' % (loss_np, smooth_loss))
    return train_loss


def val_epoch(model, loader, is_ext=None, n_test=1, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []

    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device), target.to(device)
            logits = model(data)
            LOGITS.append(logits.detach().cpu())
            TARGETS.append(target.detach().cpu())

    val_loss = criterion(torch.cat(LOGITS).squeeze() , torch.cat(TARGETS)).numpy()
    PROBS = torch.sigmoid(torch.cat(LOGITS)).numpy().squeeze()    
    LOGITS = torch.cat(LOGITS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    
    if get_output:
        return LOGITS, PROBS, TARGETS
    else:
        acc = (PROBS.round() == TARGETS).mean() * 100.
        auc = roc_auc_score(TARGETS, LOGITS)
        return float(val_loss), acc, auc
    
def run(fold):
    df_train = meta[(meta['fold'] != fold)]
    df_valid = meta[(meta['fold'] == fold)]

    dataset_train = BMDataset3D(df_train, 'train', transform=train_transforms)
    dataset_valid = BMDataset3D(df_valid, 'val', transform=val_transforms)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, 
                                               sampler=RandomSampler(dataset_train), 
                                               num_workers=num_workers,
                                               drop_last=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size= args.bs // 4, num_workers= args.num_workers)

    model = get_model(num_classes = args.out_dim).cuda()
    
    auc_best = 0
    model_file = f'{args.name}_best_fold{fold}.pth'

    optimizer = optim.Adam(model.parameters(), lr=args.init_lr)
    
    n_device = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    
    if n_device > 1:
        print(f'running on {n_device} GPUs')
        model = nn.DataParallel(model)  # todo: ddp, sync batchnorm
        if args.pretrained_path:
            print('loading pretrained model...')
            model.load_state_dict(torch.load(args.pretrained_path))
        
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    print(len(dataset_train), len(dataset_valid))

    for epoch in range(1, args.n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, acc, auc = val_epoch(model, valid_loader)
        
        scheduler.step(auc)
        
        content = time.ctime() + ' ' + f'Fold {fold}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {np.mean(train_loss):.5f}, valid loss: {(val_loss):.5f}, acc: {(acc):.4f}, auc: {(auc):.6f}'
        print(content)
        with open(f'log_{args.name}.txt', 'a') as appender:
            appender.write(content + '\n')             
            
        if auc >  auc_best:
            print('best_auc ({:.6f} --> {:.6f}).  Saving model ...'.format(auc_best, auc))
            torch.save(model.state_dict(), model_file)
            auc_best = auc
            
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    # print args
    print('#' * 80)
    print('\nConfig:')
    for key in sorted(args.__dict__):
        print('  {:12s} {}'.format(key + ':', args.__dict__[key]))
    print('#' * 80)
    
    test_id = np.load('test_id.npy')
    meta = pd.read_csv('mask_vol_all.csv')
    meta = meta[~meta['PiCare MetID'].isin(test_id)].reset_index(drop=True)
    meta = meta.sample(frac=args.percentage).reset_index(drop=True)
    print(f'Training with {args.percentage*100}% of the data.')
    
    meta['fold'] = -1
    skf = StratifiedKFold(random_state=42, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(meta, meta['Failure-binary'])):
        meta.loc[valid_idx, 'fold'] = fold
    print('Fold splitted.')
    
    train_transforms = Compose([ScaleIntensity(), 
                            Resize((50, args.image_size, args.image_size)), 
                            RandAffine( 
                                      prob= 0.5,
                                      translate_range=(1, 5, 5),
                                      rotate_range=(0.2, 0.2, 0.2), 
                                      scale_range=(0, 0.1, 0.1),
                                      padding_mode="border"),
                            ToTensor()])

    val_transforms = Compose([ScaleIntensity(), Resize((50, args.image_size, args.image_size)), ToTensor()]) 
    criterion = nn.BCEWithLogitsLoss()

    run(0)
    run(1)
    run(2)
    run(3)
    run(4)
    
  main()
