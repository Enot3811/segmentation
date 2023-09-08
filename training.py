from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
import numpy as np
from tqdm import tqdm

from segmentation_dataset import SegmentationDataset
from model_utils import (
    get_model, create_segmentation_map, overlay_segmentation)
from metrics import segmentation_iou
from dataset_parameters.original_parameters import (
    n_classes, cls_weights, cls_names, label_colors)
from numpy_utils import tensor_to_numpy


def main():
    # Data parameters
    train_imgs = Path(
        '/home/pc0/projects/segmentation/data/ign/images/training')
    val_imgs = Path(
        '/home/pc0/projects/segmentation/data/ign/images/validation')
    train_masks = Path(
        '/home/pc0/projects/segmentation/data/ign/annotations/training')
    val_masks = Path(
        '/home/pc0/projects/segmentation/data/ign/annotations/validation')
    crop_size = (513, 513)
    batch_size = 16
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # Train parameters
    device = 'cuda'
    epochs = 400
    lr = 0.001
    weights_decay = 1e-2
    scheduler_gamma = 0.99
    work_dir = 'work_dir/train_4'
    continue_training = False

    # Prepare some stuff
    device = torch.device(device)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    # name2index = {name: i for i, name in enumerate(cls_names)}
    index2name = {i: name for i, name in enumerate(cls_names)}
    mean = np.array(mean)
    std = np.array(std)

    with open(work_dir / 'train_info.txt', 'w') as f:
        f.write(f'lr {lr}\nl2 {weights_decay}\nb_size {batch_size}')

    if continue_training:
        checkpoint = torch.load(work_dir / 'last_checkpoint.pth')
        model_params = checkpoint['model']
        optim_params = checkpoint['optimizer']
        start_ep = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
    else:
        model_params = None
        optim_params = None
        start_ep = 0
        best_loss = None

    # Get tensorboard
    log_writer = SummaryWriter(str(work_dir / 'tensorboard'))

    # Get augmentations
    transforms = A.Compose([
        A.VerticalFlip(),
        A.HorizontalFlip(),
        A.RandomRotate90(p=1.0),
        A.Transpose(),
        A.OneOf([
            A.Resize(*crop_size),
            A.RandomResizedCrop(*crop_size)
        ], p=1.0),
        A.Normalize(mean, std),
        ToTensor()
    ])

    # Get datasets and data loaders
    train_dset = SegmentationDataset(train_imgs, train_masks, transforms)
    val_dset = SegmentationDataset(val_imgs, val_masks, transforms)
    train_loader = DataLoader(train_dset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dset, batch_size)

    # Get the model
    model = get_model(n_classes, pretrained=True).to(device=device)
    if model_params:
        model.load_state_dict(model_params)

    # Get the loss
    loss_func = torch.nn.CrossEntropyLoss(
        torch.tensor(cls_weights, device=device))

    # Get the optimizer
    optimizer = optim.Adam(
        model.parameters(), lr, weight_decay=weights_decay)
    if optim_params:
        optimizer.load_state_dict(optim_params)
    if start_ep == 0:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, scheduler_gamma)
    else:
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, scheduler_gamma, last_epoch=start_ep)

    # Iterate over epochs
    for e in range(start_ep, epochs):

        # Train pass
        model.train()
        train_losses = []
        aux_losses = []
        val_losses = []
        val_ious = [[] for _ in range(n_classes)]
        desc = f'Train epoch {e}'
        for batch in tqdm(train_loader, desc=desc):
            images = batch[0].to(device=device)
            masks = batch[1].to(dtype=torch.long, device=device)
            out = model(images)
            aux_predict = out['aux']
            predict = out['out']
            entropy_loss = loss_func(predict, masks)
            aux_loss = loss_func(aux_predict, masks)
            optimizer.zero_grad()
            total_loss = entropy_loss + aux_loss
            total_loss.backward()
            optimizer.step()
            train_losses.append(entropy_loss.item())
            aux_losses.append(aux_loss.item())

        # Validation pass
        with torch.no_grad():
            model.eval()
            save_imgs = True
            desc = f'Val epoch {e}'
            for batch in tqdm(val_loader, desc=desc):
                images = batch[0].to(device=device)
                masks = batch[1].to(dtype=torch.long, device=device)
                predict = model(images)['out']
                val_losses.append(loss_func(predict, masks).item())
                cls_ious = segmentation_iou(predict, masks, ignore=[0])
                for i, cls_iou in enumerate(cls_ious.tolist()):
                    val_ious[i].append(cls_iou)

                # Save an example to tensorboard
                if save_imgs:
                    segmentation = create_segmentation_map(
                        predict, label_colors)
                    save_images = tensor_to_numpy(images)
                    save_images = (
                        (save_images * std.reshape(1, 1, 1, 3) +
                         mean.reshape(1, 1, 1, 3)) * 255).astype(np.uint8)
                    for i in range(batch_size):
                        overlay_img = overlay_segmentation(
                            save_images[i], segmentation[i])
                        stacked = np.hstack(
                            (save_images[i], segmentation[i], overlay_img))
                        log_writer.add_image(
                            f'Example_{i}', stacked, e, dataformats='HWC')
                    save_imgs = False

        # Logging
        train_loss = sum(train_losses) / len(train_losses)
        train_aux_loss = sum(aux_losses) / len(aux_losses)
        val_loss = sum(val_losses) / len(val_losses)
        log_writer.add_scalar('Train_loss', train_loss, e)
        log_writer.add_scalar('Aux_loss', train_aux_loss, e)
        log_writer.add_scalar('Val_loss', val_loss, e)

        for i in range(len(val_ious)):
            val_ious[i] = sum(val_ious[i]) / len(val_ious[i])
            log_writer.add_scalar(
                f'Classes_IoU/{index2name[i]}', val_ious[i], e)
        mean_iou = sum(val_ious) / len(val_ious)
        log_writer.add_scalar('Mean_IoU', mean_iou, e)

        cur_lr = optimizer.param_groups[0]["lr"]
        log_writer.add_scalar('Learning_rate', cur_lr, e)

        print(f'Epoch: {e} '
              f'Lr: {cur_lr}'
              f'train_loss: {train_loss} '
              f'aux_loss: {aux_loss} '
              f'val_loss: {val_loss} '
              f'IoU_metric: {mean_iou}')
        
        # Lr scheduler
        lr_scheduler.step()

        # Saving
        if best_loss is None or best_loss > val_loss:
            best_loss = val_loss
            torch.save(
                model.state_dict(),
                work_dir / 'best_model.pt')

        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': e + 1,
                'best_loss': best_loss
            },
            work_dir / 'last_checkpoint.pth')

    log_writer.close()


if __name__ == '__main__':
    main()
