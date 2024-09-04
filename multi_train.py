import torch.optim as optim
import warnings
import torch
import argparse
import pandas as pd
import os
from time import time
from tqdm import tqdm
from model import SkinModel, ViTSkinModel, SkinConvNext, SkinMaxVit, SkinCoat
from multi_utils import criterion, prepare_loaders, df_preprocess, set_seed, get_efficient_model_list, get_vit_model_list
import gc
from colorama import Fore, Back, Style
from torcheval.metrics.functional import binary_auroc
from inference import inference
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch.multiprocessing as mp
import timm
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

def train_model(model, optimizer, scheduler, dataloader, epoch, train_sampler, local_rank):
    
    model.train()
    train_sampler.set_epoch(epoch)
    
    dataset_size = 0
    running_loss = 0.0
    running_auroc = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(local_rank, dtype=torch.float32)
        targets = data['target'].to(local_rank, dtype=torch.float32)
        
        batch_size = images.size(0)
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        
        # gradient acculmuation 사용 시 코드 추가할 것
        # loss = loss / grad_accul
        
        loss.backward()
        
        optimizer.step()
        
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


def full_train(args):
    BASE_DIR = os.path.dirname(__file__)
    model_dir_path = f'{BASE_DIR}/{args.model_name}'
    
    torch.cuda.empty_cache()
    
    df = df_preprocess(nfold=args.n_fold, fold=args.fold)
    
    args.global_rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    
    torch.cuda.set_device(args.local_rank)
    
    if args.global_rank is not None and args.local_rank is not None:
        print('Use GPU [{}/{}] for training'.format(args.global_rank, args.local_rank))
    
    
    dist.init_process_group(backend='nccl')
    
    dist.barrier()
    
    # num_worker 문제 방지를 위해 데이터로더를 먼저 불러온다.
    num_workers = torch.cuda.device_count() * 4
    train_loader, valid_loader, train_sampler = prepare_loaders(df, args.t_batch_size, args.v_batch_size, args.img_size, 
                                                 num_workers, args.world_size, args.fold)
    
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
    elif args.model_name in timm.list_models('**coat**', pretrained=True):
        model = SkinCoat(model_name=args.model_name, pretrained=True, checkpoint_path=None)
    
    
    model.cuda(args.local_rank)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=1) # 출력은 하나로 모으기
    
    
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
    
    
    num_epoch = args.num_epoch
    
    
    start = time()
    best_model_wts = deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf
    history = defaultdict(list)
    best_record = defaultdict(list)
    
    # Make a directory for saved model
    if not os.path.exists(model_dir_path):
        os.mkdir(model_dir_path)
    
    for epoch in range(1, num_epoch + 1):
        gc.collect()
        
        train_loss, train_auc = train_model(model, optimizer, scheduler, train_loader, epoch, train_sampler, args.local_rank)
        val_loss, val_auc = inference(model, optimizer, valid_loader, epoch, args.local_rank)
        
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
            
            torch.save(best_model_wts, model_save_path)
            
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
    

    
if __name__ == '__main__':
    # base directory
    if torch.cuda.is_available():
        print(f'Using GPU : {torch.cuda.get_device_name()}')
        
    warnings.filterwarnings("ignore")
    
    # For colored terminal text
    b_ = Fore.BLUE
    sr_ = Style.RESET_ALL
            
    # For descriptive error messages
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    
    parser = argparse.ArgumentParser()
    
    ## argument 추가
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--global_rank', type=int, default=0)
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
    
    full_train(args)


    
    
    
    
    
    