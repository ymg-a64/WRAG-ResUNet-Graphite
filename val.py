
import torch
from torch.utils.data import DataLoader
import time

from model.unet_resnet import Unet
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.train_and_eval import pixel_accuracy, mean_accuracy, mean_iou, frequency_weighted_iou

class LogColor:
    GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    RESET = "\033[0m"
    BLUE = "\033[1;34m"

def evaluate(model, val_loader, device, num_classes):
    model_eval = model.eval()
    model_eval = model_eval.cuda()

    total_pixel_acc = 0
    total_mean_acc = 0
    total_mean_iou = 0
    total_fw_iou = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        for iteration, batch in enumerate(val_loader):
            imgs, pngs, labels = batch
            imgs = imgs.to(device)
            pngs = pngs.to(device)
            outputs = model_eval(imgs)

            pixel_acc = pixel_accuracy(outputs, pngs)
            mean_acc = mean_accuracy(outputs, pngs, num_classes)
            mean_iou_value = mean_iou(outputs, pngs, num_classes)
            fw_iou = frequency_weighted_iou(outputs, pngs, num_classes)

            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_iou += mean_iou_value
            total_fw_iou += fw_iou

            if iteration == 0:
                data_num_len = len("data_num") - len("data_num") + 12
                Pixelacc_len = len("GPU Mem") - len("Pixelacc") + 12
                Meanacc_len = len("Loss") - len("Meanacc") + 12
                Meaniou_len = len("LR") - len("Meaniou") + 12

                print(
                      f"{LogColor.RED}data_num{LogColor.RESET}{' ' * data_num_len}"
                      f"{LogColor.RED}Pixelacc{LogColor.RESET}{' ' * Pixelacc_len}"
                      f"{LogColor.RED}Meanacc{LogColor.RESET}{' ' * Meanacc_len}"
                      f"{LogColor.RED}Meaniou{LogColor.RESET}{' ' * Meaniou_len}"
                      f"{LogColor.RED}Fwiou{LogColor.RESET}")

    avg_pixel_acc = total_pixel_acc / num_batches
    avg_mean_acc = total_mean_acc / num_batches
    avg_mean_iou = total_mean_iou / num_batches
    avg_fw_iou = total_fw_iou / num_batches


    batch_len = data_num_len + len("data_num") - len(str(f"{len(val_loader.dataset)}"))
    avg_pixel_acc_len = Pixelacc_len + len("Pixelacc") - len(str(f"{avg_pixel_acc:.2f}"))
    avg_mean_acc_len = Meanacc_len + len("Meanacc") - len(str(f"{avg_mean_acc:.2f}"))
    avg_Mean_iou_len = Meaniou_len + len("Meaniou") - len(str(f"{avg_mean_iou:.2f}"))

    print(
          f"{len(val_loader.dataset)}{' ' * batch_len}"
          f"{avg_pixel_acc:.2f}{' ' * avg_pixel_acc_len}"
          f"{avg_mean_acc:.2f}{' ' * avg_mean_acc_len}"
          f"{avg_mean_iou:.2f}{' ' * avg_Mean_iou_len}"
          f"{avg_fw_iou:.2f}", end='', flush=True)
    print(f"\n{LogColor.GREEN}")
    time.sleep(1)

def val(args):
    num_classes = args.num_classes + 1
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    input_shape = [512, 512]
    val_dataset = UnetDataset(args.data_path, input_shape, num_classes, augmentation=False, txt_name="test.txt")

    val_loader = DataLoader(val_dataset,
                        shuffle=True,
                        batch_size=1,
                        num_workers=0,
                        pin_memory=True,
                        drop_last=False, collate_fn=unet_dataset_collate, sampler=None)

    model = Unet(num_classes=num_classes)
    weights_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(weights_dict)
    model.to(device)
    evaluate(model, val_loader, device, num_classes)



def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch fcn training")
    parser.add_argument("--data-path", default="VOCdevkit-1", help="VOCdevkit-1 root")
    parser.add_argument("--weights", default="run/train/exp1/weights/best_model_1.pth")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--device", default="cuda", help="training device")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    val(args)
