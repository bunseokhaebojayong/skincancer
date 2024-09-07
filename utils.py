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

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, 
                              shuffle=True, pin_memory=True, drop_last=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=val_batch_size,
                              shuffle=False, pin_memory=True, num_workers=4)
    
    return train_loader, valid_loader

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