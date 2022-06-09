import os
import torch
import argparse

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets

from loguru import logger
from torchvision import transforms as T
from utils import train, validate, get_dataloader, get_device, make_dir


def make_args():
    parser = argparse.ArgumentParser(description="CIFAR10--FineTune")
    # --------------------configurations of dataset -------------------------------
    parser.add_argument("--download", type=str, default=False,
                        help="Enable download dataset")
    parser.add_argument("--valid-split", default=0.2,
                        help="Split ratio of training dataset")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="input batch size for training")
    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N",
                        help="input batch size for testing")

    # ---------------------configurations of training------------------------------
    parser.add_argument("--epochs", type=int, default=100, metavar="N",
                        help="number of epochs to train")
    parser.add_argument("--ImageAugmentation", action="store_true",
                        help="Enable image augmentation")

    # ---------------------configurations of learning rate scheduler -----------------
    parser.add_argument("--init-lr", type=float, default=0.1,
                        help="initial learning rate")

    # ---------------------configurations of saving------------------------------
    parser.add_argument("--ckpts-dir", type=str, default="./ckpts",
                        help="directory to save checkpoints")
    parser.add_argument("--imgs-dir", type=str, default="./imgs",
                        help="directory to save images")
    parser.add_argument("--logs-dir", type=str, default="./logs",
                        help="directory to save log file")
    parser.add_argument("--dataset-root", type=str, default="./data/CIFAR10",
                        help="root to datasets")

    # ---------------------configurations of backbone ------------------------------
    parser.add_argument("--backbone-name", type=str, default=None,
                        choices=["vgg", "resnet"], help="backbone name")
    parser.add_argument("--freeze-backbone", type=str, default=None,
                        action="store-true", help="enable freeze backbone")
    return parser


def get_prefix(args)->str:
    list_str = [args.backbone_name, args.freeze_back_bone]
    return "_".join(list_str)


def get_CIFAR10loader(args):
    list_transforms = [
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if args.ImageAugmentation:
        augmenter = [
            T.RandomHorizontalFlip(),
            T.RandomCrop(size=32, padding=2),
        ]
        transform=T.Compose(augmenter + list_transforms)
    else:
        transform=T.Compose(list_transforms)
    test_transform = T.Compose(list_transforms)
    train_dataset = datasets.CIFAR10(
        root=args.dataset_root,
        train=True,
        transform=transform,
        download=args.download,
    )
    test_dataset = datasets.CIFAR10(
        root=args.dataset_root,
        train=False,
        transform=test_transform,
        download=args.download,
    )
    train_loader, val_loader, test_loader = get_dataloader(
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_workers=4,
        valid_split=args.valid_split,
    )
    return train_loader, val_loader, test_loader


def fit(
    model: nn.Module,
    crit,
    epochs,
    init_lr,
    ckpt_file,
    train_loader,
    val_loader,
):
    device = get_device()
    model = model.to(device)
    train_loss, train_acc, val_loss, val_acc = [[] for _ in range(4)]
    optimizer = optim.SGD(model.parameters(), lr=init_lr)
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

        logger.info(f"[INFO]Train Epoch: {epoch} / {epochs} LR: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Train Loss: {train_loss_:.5f}\tTrain Accuracy: {train_acc_:.5f}")
        logger.info(f"Valid Loss: {val_loss_:.5f}\tValid Accuracy: {val_acc_:.5f}\n")

    logger.info(f"Saving model to {ckpt_file}\n")
    torch.save(model.state_dict(), ckpt_file)
    history = {
        "train_history": {"train_accuracy": train_acc, "train_loss": train_loss},
        "val_history": {"val_accuracy": val_acc, "val_loss": val_loss},
    }
    return history


class M(nn.Module):
    def __init__(
        self,
        backbone_name,
        freeze_backbone,
        num_classes,
    ):
        super(M, self).__init__()
        if backbone_name == "vgg":
            self.backbone = models.vgg16(pretrained=True)
        elif backbone_name == "resnet":
            self.backbone = models.resnet18(pretrained=True)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        logits = self.classifier(x)
        return logits

def main():
    args = make_args().parse_args()
    make_dir(args)
    prefix = get_prefix(args)
    log_file = os.path.join(args.logs_dir, prefix+"_{time}.log")
    logger.add(log_file)
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument {}: {}", arg, value)

    num_classes = 10
    model = M(
        backbone_name=args.backbone_name,
        freeze_backbone=args.freeze_backbone,
        num_classes=num_classes,
    )
    logger.info("model:\n {}", model)

    ckpt_file = os.path.join(args.ckpts_dir, prefix+".pth")
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = get_CIFAR10loader(args)
    if args.ImageAugmentation:
        logger.info("Using ImageAugmentation.")
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
