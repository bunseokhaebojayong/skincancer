import torch.optim as optim
import warnings
import torch
import argparse
import pandas as pd
import os
from time import time
from tqdm import tqdm
from model import SkinModel, ViTSkinModel, SkinConvNext, SkinMaxVit, SkinCoat
from utils import criterion, prepare_loaders, df_preprocess, set_seed, get_efficient_model_list, get_vit_model_list
import gc
from colorama import Fore, Back, Style
from torcheval.metrics.functional import binary_auroc
from inference import inference
from collections import defaultdict
from copy import deepcopy
import numpy as np
import timm

def train_model(model, optimizer, scheduler, dataloader, epoch, n_accumulate=1):
    
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to('cuda', dtype=torch.float)
        targets = data['target'].to('cuda', dtype=torch.float)
        
        batch_size = images.size(0)
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        loss = loss / n_accumulate
        
        loss.backward()
        
        if (step+1) % n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()
        
        running_loss += (loss.item() * batch_size)
        running_auroc += (auroc * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
    
    gc.collect() # garbarge collector
    bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Auroc=epoch_auroc,LR=optimizer.param_groups[0]['lr'])
    
    return epoch_loss, epoch_auroc


    
if __name__ == '__main__':
    # base directory
    BASE_DIR = os.path.dirname(__file__)

    warnings.filterwarnings("ignore")
    
    # For colored terminal text
    b_ = Fore.BLUE
    sr_ = Style.RESET_ALL
    

    # checkpoint_path = './checkpoints'
    # if not os.path.exists(checkpoint_path):
    #     os.mkdir(checkpoint_path)
        
    # For descriptive error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    parser = argparse.ArgumentParser()
    
    ## argument 추가
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--model_name', type=str, help='돌리고자 하는 모델명입니다.')
    parser.add_argument('--img_size', type=int, help='image_size')
    parser.add_argument('--num_epoch', type=int, help='epoch steps')
    parser.add_argument('--t_batch_size', type=int, default=32, help='Train batch size')
    parser.add_argument('--v_batch_size', type=int, default=64, help='valid batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_scheduler', default='cosine', help='learning rate scheduler')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='eta_min')
    parser.add_argument('--t_max', default=500, type=int, help='t_max')
    parser.add_argument('--t_0', type=float, help='t_0')
    parser.add_argument('--w_decay', default=1e-6, type=float, help='weight decay')
    parser.add_argument('--fold', default=0, type=int, help='fold')
    parser.add_argument('--n_fold', default=5, type=int, help='n_fold')
    parser.add_argument('--n_accumulate', default=1, type=int, help='n_accumulate')
    
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    model_dir_path = f'{BASE_DIR}/{args.model_name}'
    
    df = df_preprocess(nfold=args.n_fold, fold=args.fold)
    
    if args.model_name in get_vit_model_list():
        model = ViTSkinModel(model_name=args.model_name, pretrained=True, 
                      checkpoint_path=None)
    elif args.model_name in get_efficient_model_list():
        model = SkinModel(model_name=args.model_name, pretrained=True, 
                      checkpoint_path=None)
    elif args.model_name in timm.list_models('*convnext*', pretrained=True):
        model = SkinConvNext(model_name=args.model_name, pretrained=True, checkpoint_path=None)
    elif args.model_name in timm.list_models('**maxvit**', pretrained=True):
        model = SkinMaxVit(model_name=args.model_name, pretrained=True, checkpoint_path=None)
    elif args.model_name in timm.list_models('**coat**', pretrained=False):
        model = SkinCoat(model_name=args.model_name, pretrained=False, checkpoint_path=None)
    
    model.to('cuda')
    
    if torch.cuda.is_available():
        print(f'Using GPU : {torch.cuda.get_device_name()}')
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.w_decay)

    # scheduler 설정
    def set_lr_scheduler(lr_scheduler, optimizer):
        if lr_scheduler == 'cosine':
            T_max = args.t_max
            min_lr = args.min_lr
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, min_lr)
        elif lr_scheduler == 'cosine-warm':
            T_0 = args.t_0
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0)
        return scheduler
    scheduler = set_lr_scheduler(args.lr_scheduler, optimizer)
    
    train_loader, valid_loader = prepare_loaders(df, args.t_batch_size, args.v_batch_size, args.img_size, args.fold)
    num_epoch = args.num_epoch
    
    
    start = time()
    best_model_wts = deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    best_record = defaultdict(list)
    
    n_accumulate = args.n_acumulate
    
    # Make a directory for saved model
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    
    for epoch in range(1, num_epoch + 1):
        gc.collect()
        
        train_loss, train_auc = train_model(model, optimizer, scheduler, train_loader, epoch, n_accumulate)
        val_loss, val_auc = inference(model, optimizer, valid_loader, epoch)
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)
        history['lr'].append(scheduler.get_lr()[0])
        
        # deep copy the model
        if best_epoch_auroc <= val_auc:
            print(f"{b_}Validation AUROC Improved ({best_epoch_auroc} ---> {val_auc})")
            best_epoch_auroc = val_auc
            best_model_wts = deepcopy(model.state_dict())
            
            
            # Record Best record! 
            best_record['model_name'].append(args.model_name)
            best_record['epoch'].append(epoch)
            best_record['val_loss'].append(val_loss)
            best_record['val_auc'].append(val_auc)
            model_save_path = f'{model_dir_path}/auc{val_auc}_loss{val_loss}_epoch{epoch}.bin'
            
            torch.save(model.state_dict(), model_save_path)
            
            # Save a model file from the current directory
            print(f'Model Saved!{sr_}')
            
        print()
        
    end = time()
    time_elapsed = end - start
    
    # Write the best record from the model
    best_record_csv = pd.DataFrame.from_dict(best_record)
    best_record_csv.to_csv("best_record.csv", mode='a', index=False)
    
    print(f'Train has completed in {(time_elapsed // 3600):.0f}h:{((time_elapsed % 3600) // 60):.0f}m:{((time_elapsed % 3600) % 60):.0f}s')
    print(f'Best AUROC : {best_epoch_auroc:.4f}')
    
    history = pd.DataFrame.from_dict(history)
    history.to_csv("history.csv", index=False)
    




    
    
    
    
    
    