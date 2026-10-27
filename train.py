
import os
from functools import partial
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import datetime
import subprocess


from model.unet_resnet import Unet
from model.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, worker_init_fn
from utils.train_and_eval import train_one_epoch, evaluate

from utils.create_exp_folder import create_exp_folder
from utils.plot_results import plot_training_curves


def get_gpu_usage():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    used, total = map(int, result.strip().split(','))

    return used


def create_model(num_classes, weights):
    model = Unet(num_classes=num_classes)
    weights_init(model)

    if weights:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(weights, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
    return model



def get_optimizer_and_lr(model, batch_size, train_epoch, momentum, weight_decay):
    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    lr_decay_type = 'cos'
    nbs = 16
    lr_limit_max = 1e-4
    lr_limit_min = 1e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999),
                           weight_decay=weight_decay)
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, train_epoch)
    return optimizer, lr_scheduler_func


def train(args):
    seed_everything(11)

    num_classes = args.num_classes + 1
    train_epoch = args.epochs
    batch_size = args.batch_size
    num_workers = args.workers
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    exp_folder, weights_folder = create_exp_folder()

    input_shape = [512, 512]

    train_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=True, txt_name="train.txt")
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=False, txt_name="val.txt")

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False,
                              collate_fn=unet_dataset_collate,
                              sampler=None,
                              worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    val_loader = DataLoader(val_dataset,
                            shuffle=True,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=True,
                            drop_last=False,
                            collate_fn=unet_dataset_collate,
                            sampler=None,
                            worker_init_fn=partial(worker_init_fn, rank=0, seed=11))

    model = create_model(num_classes=num_classes, weights=args.weights)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    optimizer, lr_scheduler_func = get_optimizer_and_lr(model, batch_size, train_epoch,  args.momentum, args.weight_decay)
    start_time = time.time()
    best_acc = 0.0
    best_model_path = os.path.join(weights_folder, f"best_model_{args.num_classes}.pth")
    last_model_path = os.path.join(weights_folder, f"last_model_{args.num_classes}.pth")

    train_losses = []
    val_losses = []
    val_metrics_history = []

    focal_loss = False
    dice_loss = True

    for epoch in range(train_epoch):
        gpu_used = get_gpu_usage()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        loss = train_one_epoch(model, optimizer, train_loader, device, dice_loss, focal_loss,
                               gpu_used, num_classes, scaler, epoch, train_epoch)

        train_losses.append(loss)
        metrics = evaluate(model, val_loader, device, dice_loss, focal_loss, num_classes)

        val_losses.append(metrics["Loss"])
        val_metrics_history.append(metrics)

        current_acc = float(metrics["Mean Accuracy"])
        if current_acc > best_acc:
            best_acc = current_acc
            torch.save(model.state_dict(), best_model_path)

        torch.save(model.state_dict(), last_model_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))

    plot_training_curves(train_losses, val_losses, val_metrics_history, weights_folder)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--weights", default="weights/unet_resnet_voc.pth",
                        help="Path to the directory containing model weights")
    parser.add_argument("--data-path", default="VOCdevkit-1", help="VOCdevkit-1 root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int, metavar="N", help="number of total epochs to train")
    parser.add_argument("--workers", default=0, type=int, metavar="N",
                        help="number of data loading workers (default: 0, meaning data loading runs in main process)")
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=True, type=bool, help="Use torch.cuda.amp for mixed precision training")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)