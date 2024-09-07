import torch
from utils import criterion
from torcheval.metrics.functional import binary_auroc
from tqdm import tqdm
import gc

@torch.inference_mode()
def inference(model, optimizer, dataloader, epoch):
    model.eval()

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

        auroc = binary_auroc(input=outputs.squeeze(), target=targets).item()

        running_loss += (loss.item() * batch_size)
        running_auroc += (auroc * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_auroc = running_auroc / dataset_size
        
        bar.set_postfix(epoch=epoch, val_loss=epoch_loss, val_auroc=epoch_auroc,LR=optimizer.param_groups[0]['lr'])

    gc.collect() # garbarge collector

    return epoch_loss, epoch_auroc