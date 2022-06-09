import os
import torch
import argparse

import torch.nn as nn
import torch.optim as optim

from loguru import logger
from models.cnn import LeNet_3Channel
from torchvision import transforms as T
from tools.checkpoints import ModelSaver
from utils import train, validate, get_CIFAR10loader, make_dir, get_device, adjust_lr


def make_args():
    parser = argparse.ArgumentParser(description="CIFAR10--LeNet")
    # --------------------configurations of dataset -------------------------------
    parser.add_argument("--download", type=str, default=False,
                        help="Enable download dataset")
    parser.add_argument("--valid-split", default=0.2,
                        help="Split ratio of training dataset")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1024, metavar="N",
                        help="input batch size for testing (default: 1024)")

    # ---------------------configurations of training------------------------------
    parser.add_argument("--epochs", type=int, default=60, metavar="N",
                        help="number of epochs to train")
    parser.add_argument("--ImageAugmentation", action="store_true",
                        help="Enable Image Augmentation.")

    # ---------------------configurations of learning rate scheduler -----------------
    parser.add_argument("--init-lr", type=float, default=0.1,
                        help="initial learning rate")
    parser.add_argument("--LRScheduler", action="store_true",
                        help="Enable learning rate scheduler")
    parser.add_argument("--milestone1", type=int, default=20,
                        help="milestome of lr scheduler")
    parser.add_argument("--milestone2", type=int, default=35,
                        help="milestome of lr scheduler")   

    # ---------------------configurations of saving------------------------------
    parser.add_argument("--ckpts-dir", type=str, default="./ckpts",
                        help="directory to save checkpoints")
    parser.add_argument("--imgs-dir", type=str, default="./imgs",
                        help="directory to save images")
    parser.add_argument("--logs-dir", type=str, default="./logs",
                        help="directory to save log file")
    parser.add_argument("--dataset-root", type=str, default="./data/CIFAR10/",
                        help="root to datasets")
    return parser


def get_prefix()->str:
    return "LeNet_CIFAR10"


def get_augmenter(image_augmentation):
    if not image_augmentation:
        return None
    logger.info("Using image augmentation.")

    # augmenter = [T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)]
    
    augmenter = [T.ColorJitter(contrast=0.3)]
    
    # augmenter = [T.RandomPerspective(distortion_scale=0.4, p=0.4)]
    
    # augmenter = [T.RandomAffine(degrees=(0, 15), translate=(0, 0.1), scale=(0.75, 0.95))]

    '''
    augmenter = [
        T.RandomPerspective(distortion_scale=0.4, p=0.4),
        T.ColorJitter(brightness=.5, hue=.3),
    ]
    '''

    logger.info(f"{augmenter}\n")
    return augmenter


def fit(
    model: nn.Module,
    crit,
    epochs,
    init_lr,
    ckpt_file,
    train_loader,
    val_loader,
    use_lrscheduler,
    milestone1,
    milestone2,
):
    device = get_device()
    model = model.to(device)
    if use_lrscheduler:
        logger.info("Using learning rate scheduler.\n")
    train_loss, train_acc, val_loss, val_acc = [[] for _ in range(4)]
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
    modelsaver = ModelSaver(mode="max", ckpt_file=ckpt_file)
    for epoch in range(1, epochs+1):
        # training
        train_loss_, train_acc_ = train(
            model=model,
            device=device,
            criterion=crit,
            optimizer=optimizer,
            train_loader=train_loader,
        )
        train_loss.append(train_loss_)
        train_acc.append(train_acc_)

        # validation
        val_loss_, val_acc_ = validate(
            model=model,
            device=device,
            criterion=crit,
            val_loader=val_loader,
        )
        val_loss.append(val_loss_)
        val_acc.append(val_acc_)

        logger.info(f"Train Epoch: {epoch} / {epochs} LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Train Loss: {train_loss_:.5f}\tTrain Accuracy: {train_acc_:.5f}")
        logger.info(f"Valid Loss: {val_loss_:.5f}\tValid Accuracy: {val_acc_:.5f}\n")
        modelsaver(val=val_acc_, model=model)
        if use_lrscheduler:
            adjust_lr(
                optimizer=optimizer,
                epoch=epoch,
                milestone1=milestone1,
                milestone2=milestone2,
            )
    history = {
        "train_history": {"train_accuracy": train_acc, "train_loss": train_loss},
        "val_history": {"val_accuracy": val_acc, "val_loss": val_loss},
    }
    return history


def main():
    args = make_args().parse_args()
    make_dir(args)
    prefix = get_prefix()
    log_file = os.path.join(args.logs_dir, prefix+"_{time}.log")
    logger.add(log_file)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}", arg, value)

    num_classes = 10
    model = LeNet_3Channel(num_classes=num_classes)
    logger.info("model:\n {}", model)

    ckpt_file = os.path.join(args.ckpts_dir, prefix+".pth")
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = get_CIFAR10loader(
        download=args.download,
        root=args.dataset_root,
        valid_split=args.valid_split,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        augmenter=get_augmenter(args.ImageAugmentation),
    )
    history = fit(
        model=model,
        crit=crit,
        epochs=args.epochs,
        init_lr=args.init_lr,
        ckpt_file=ckpt_file,
        train_loader=train_loader,
        val_loader=val_loader,
        use_lrscheduler=args.LRScheduler,
        milestone1=args.milestone1,
        milestone2=args.milestone2,
    )
    model.load_state_dict(torch.load(ckpt_file))
    test_loss, test_acc = validate(
        model=model,
        device=get_device(),
        criterion=crit,
        val_loader=test_loader,
    )
    logger.info(f"Test Loss: {test_loss:.5f}\tTest Accuracy: {test_acc:.5f}")
    history["test"] = {"test_loss": test_loss, "test_acc": test_acc}
    logger.info("History:{}\n", history)
    
    logger.info("Saving log to {}", log_file)
    logger.info("Finish!")


if __name__ == "__main__":
    main()
