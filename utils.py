import random
import numpy as np
import torch
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from pathlib import Path
from glob import glob
from torch.utils.data import DataLoader
from glob import glob

class SkinDataSet(Dataset):
    def __init__(self, df, cat, transforms=None):
        self.cat = cat
        if cat == 'train':
            self.df_positive = df[df['target'] == 1].reset_index()
            self.df_negative = df[df['target'] == 0].reset_index()
            self.file_names_positive = self.df_positive['file_path'].values
            self.file_names_negative = self.df_negative['file_path'].values
            self.targets_positive = self.df_positive['target'].values
            self.targets_negative = self.df_negative['target'].values
            self.transforms = transforms
        else:
            self.df = df
            self.file_names = df['file_path'].values
            self.targets = df['target'].values
            self.transforms = transforms
        
    def __len__(self):
        if self.cat == 'train':
            return len(self.df_positive) * 2
        else:
            return len(self.df)
    
    def __getitem__(self, index):
        if self.cat == 'train':
            if random.random() >= 0.5:
                df = self.df_positive
                file_names = self.file_names_positive
                targets = self.targets_positive
            else:
                df = self.df_negative
                file_names = self.file_names_negative
                targets = self.targets_negative
        
            index = index % df.shape[0]
            img_path = file_names[index]
            target = targets[index]
        else:
            img_path = self.file_names[index]
            target = self.targets[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        return {'image': img, 'target': target}
    
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)



def prepare_loaders(df, train_batch_size, val_batch_size, img_size, fold=0):
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    data_transforms = {
    "train": A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=60, 
                           p=0.5),
        A.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
    }
    
    train_dataset = SkinDataSet(df_train, cat='train', transforms=data_transforms['train'])
    valid_dataset = SkinDataSet(df_valid, cat='valid', transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, 
                              shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size,
                              shuffle=False, pin_memory=True, num_workers=4)
    
    return train_loader, valid_loader

def df_preprocess(nfold, fold):
    path = Path('../data')
    train_img_path = path /'train-image/image'
    train_csv = pd.read_csv(path / 'train-metadata.csv', low_memory=False)
    
    def get_img_path(image_id):
        return f"{train_img_path}/{image_id}.jpg"

    # Count the Images
    train_img_path = path / 'train-image/image'
    train_images = glob(str(train_img_path) + '/*')

    df_positive = train_csv[train_csv["target"] == 1].reset_index(drop=True)
    df_negative = train_csv[train_csv["target"] == 0].reset_index(drop=True)

    print(f"Total num of train_img set: {len(train_images)}")
    print(f'Train metadata shape : {train_csv.shape}')
    print(f'sum of the targets: {train_csv.target.sum()}')
    print(f'{train_csv["patient_id"].unique().shape}')
    print(df_positive.shape)
    print(df_negative.shape)

    # 비율 맞추기(baseline 참고시 1: 20으로 했는데 좀 아이디어 있으면 고치시면 됩니다!)
    # 데이터 불균형 존재
    df = pd.concat([df_positive, df_negative.iloc[:df_positive.shape[0]*20, :]])
    print("filtered>", df.shape, df.target.sum(), df["patient_id"].unique().shape)
    df['file_path'] = df['isic_id'].apply(get_img_path)
    df = df[ df["file_path"].isin(train_images) ].reset_index(drop=True)
    
    sgkf = StratifiedGroupKFold(n_splits=nfold)

    for fold, ( _, val_) in enumerate(sgkf.split(df, df.target,df.patient_id)):
        df.loc[val_ , "kfold"] = int(fold)
    return df
        

def get_efficient_model_list():
    return [
'efficientnet_b0',
 'efficientnet_b0_g8_gn',
 'efficientnet_b0_g16_evos',
 'efficientnet_b0_gn',
 'efficientnet_b1',
 'efficientnet_b1_pruned',
 'efficientnet_b2',
 'efficientnet_b2_pruned',
 'efficientnet_b3',
 'efficientnet_b3_g8_gn',
 'efficientnet_b3_gn',
 'efficientnet_b3_pruned',
 'efficientnet_b4',
 'efficientnet_b5',
 'efficientnet_b6',
 'efficientnet_b7',
 'efficientnet_b8',
 'efficientnet_blur_b0',
 'efficientnet_cc_b0_4e',
 'efficientnet_cc_b0_8e',
 'efficientnet_cc_b1_8e',
 'efficientnet_el',
 'efficientnet_el_pruned',
 'efficientnet_em',
 'efficientnet_es',
 'efficientnet_es_pruned',
 'efficientnet_h_b5',
 'efficientnet_l2',
 'efficientnet_lite0',
 'efficientnet_lite1',
 'efficientnet_lite2',
 'efficientnet_lite3',
 'efficientnet_lite4',
 'efficientnet_x_b3',
 'efficientnet_x_b5',
 'efficientnetv2_l',
 'efficientnetv2_m',
 'efficientnetv2_rw_m',
 'efficientnetv2_rw_s',
 'efficientnetv2_rw_t',
 'efficientnetv2_s',
 'efficientnetv2_xl',
 'gc_efficientnetv2_rw_t',
 'test_efficientnet',
 'tf_efficientnet_b0',
 'tf_efficientnet_b1',
 'tf_efficientnet_b2',
 'tf_efficientnet_b3',
 'tf_efficientnet_b4',
 'tf_efficientnet_b5',
 'tf_efficientnet_b6',
 'tf_efficientnet_b7',
 'tf_efficientnet_b8',
 'tf_efficientnet_cc_b0_4e',
 'tf_efficientnet_cc_b0_8e',
 'tf_efficientnet_cc_b1_8e',
 'tf_efficientnet_el',
 'tf_efficientnet_em',
 'tf_efficientnet_es',
 'tf_efficientnet_l2',
 'tf_efficientnet_lite0',
 'tf_efficientnet_lite1',
 'tf_efficientnet_lite2',
 'tf_efficientnet_lite3',
 'tf_efficientnet_lite4',
 'tf_efficientnetv2_b0',
 'tf_efficientnetv2_b1',
 'tf_efficientnetv2_b2',
 'tf_efficientnetv2_b3',
 'tf_efficientnetv2_l',
 'tf_efficientnetv2_m',
 'tf_efficientnetv2_s',
 'tf_efficientnetv2_xl',
 'efficientnet_b1.ra4_e3600_r240_in1k',
 'tf_efficientnetv2_l.in21k',
 'tf_efficientnetv2_m.in21k'
]
    
def get_vit_model_list():
    return ['vit_medium_patch16_reg4_gap_256',
            'vit_mediumd_patch16_reg4_gap_384.sbb2_e200_in12k_ft_in1k',
            'vit_mediumd_patch16_reg4_gap_256.sbb2_e200_in12k_ft_in1k']