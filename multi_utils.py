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
from torch.utils.data.distributed import DistributedSampler
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
    
    def __getitem__(self, index:int):
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



def prepare_loaders(df, train_batch_size, val_batch_size, img_size, num_workers, world_size, fold=0):
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    data_transforms = {
    "train": A.Compose([
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(limit=0.2, p=0.75),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=0.7),
        A.CLAHE(clip_limit=4.0, p=0.7),
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Downscale(p=0.25),
        A.ShiftScaleRotate(shift_limit=0.1, 
                           scale_limit=0.15, 
                           rotate_limit=15,
                           border_mode=0,
                           p=0.85),
        A.HueSaturationValue(
                hue_shift_limit=10, 
                sat_shift_limit=20, 
                val_shift_limit=10, 
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
    
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=True)
    
    train_loader = DataLoader(train_dataset, shuffle=False, pin_memory=True, 
                              num_workers=int(num_workers/world_size), batch_size=int(train_batch_size/world_size)
                              ,sampler=train_sampler, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=int(val_batch_size/world_size),
                              shuffle=False, pin_memory=True, num_workers=int(num_workers/world_size)
                              ,sampler=valid_sampler)
    
    return train_loader, valid_loader, train_sampler

def df_preprocess(nfold, fold):
    path = Path('../data')
    train_img_path = path /'train-image/image'
    train_images = glob(str(train_img_path) + '/*')
    
    train = pd.read_csv(path / 'train-metadata.csv', low_memory=False)
    train_df1 = pd.read_csv(path / 'train_2019.csv', low_memory=False)
    train_df2 = pd.read_csv(path / 'train_mealanoma.csv', low_memory=False)
    
    # Count the Images
    def get_img_path(image_id):
        return f"{train_img_path}/{image_id}.jpg"

    train_df1.rename(columns={'image_name':'isic_id'}, inplace=True)
    train_df2.rename(columns={'image_name':'isic_id'}, inplace=True)


    train_df1_positive = train_df1[train_df1["target"] == 1].reset_index(drop=True)
    train_df2_positive = train_df2[train_df2['target'] == 1].reset_index(drop=True)
    train_positive = train[train['target'] == 1].reset_index(drop=True)
    train_negative = train[train["target"] == 0].reset_index(drop=True)

    train_positive_df = pd.concat([train_df1_positive, train_df2_positive, train_positive],axis=0, ignore_index=True)
    train_positive_df.isna().sum()

    df = pd.concat([train_positive_df, train_negative.iloc[:train_positive_df.shape[0] :]])

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