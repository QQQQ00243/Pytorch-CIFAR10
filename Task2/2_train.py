import os
import argparse

import torch.nn as nn

from loguru import logger
from models.cnn import LeNet
from utils import get_device, make_dir,\
    validate, fit, getFashionMNISTloader


def make_args():
    parser = argparse.ArgumentParser(description="FashionMNIST--LeNet")
    # --------------------configurations of dataset -------------------------------
    parser.add_argument("--download", type=str, default=False,
                        help="Enable download dataset")
    parser.add_argument("--valid-split", default=0.2,
                        help="Split ratio of training dataset")
    parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N",
                        help="input batch size for testing (default: 1024)")

    # ---------------------configurations of training------------------------------
    parser.add_argument("--epochs", type=int, default=20, metavar="N",
                        help="number of epochs to train")
    parser.add_argument("--init-lr", type=float, default=0.1,
                        help="initial learning rate")

    # ---------------------configurations of saving------------------------------
    parser.add_argument("--ckpts-dir", type=str, default="./ckpts",
                        help="directory to save checkpoints")
    parser.add_argument("--imgs-dir", type=str, default="./imgs",
                        help="directory to save images")
    parser.add_argument("--logs-dir", type=str, default="./logs",
                        help="directory to save log file")
    parser.add_argument("--dataset-root", type=str, default="./data",
                        help="root to datasets")
    return parser


def get_prefix()->str:
    return "LeNet_FashionMNIST"


def main():
    args = make_args().parse_args()
    make_dir(args)
    num_classes = 10
    prefix = get_prefix()
    log_file = os.path.join(args.logs_dir, prefix+"_{time}.log")
    logger.add(log_file)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}", arg, value)

    device = get_device()
    model = LeNet(num_classes)
    logger.info("model:\n {}", model)
    ckpt_file = os.path.join(args.ckpts_dir, prefix+".pth")
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = getFashionMNISTloader(
        root=args.dataset_root,
        download=args.download,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        valid_split=args.valid_split,
    )
    history = fit(
        model=model,
        crit=crit,
        epochs=args.epochs,
        init_lr=args.init_lr,
        ckpt_file=ckpt_file,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    test_loss, test_acc = validate(
        model=model,
        device=device,
        criterion=crit,
        val_loader=test_loader,
    )
    logger.info(f"Test Loss: {test_loss:.5f}\tTest Accuracy: {test_acc:.5f}\n")
    history["test"] = {"test_loss": test_loss, "test_acc": test_acc}
    logger.info("History:{}\n", history)
    
    logger.info("Saving log to {}", log_file)
    logger.info("Finish!")


if __name__ == "__main__":
    main()
