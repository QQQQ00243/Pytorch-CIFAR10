import os
import argparse

import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.datasets as datasets

from loguru import logger
from utils import get_idx_to_class
from tools.plot import show_augment


def make_args():
    parser = argparse.ArgumentParser(description="Show CIFAR10")
    parser.add_argument("--dataset-root", default="./data/CIFAR10", 
                        type=str, help="root of dataset")
    parser.add_argument("--download", default=False,
                        help="enable download")
    parser.add_argument("--imgs-dir", default="./imgs",
                        type=str, help="directory to images")
    parser.add_argument("--show", action="store_true",
                        help="show CIFAR10 examples")
    parser.add_argument("--show-augment", action="store_true",
                        help="show augmentation")                  
    return parser


def make_dir(args):
    if not os.path.exists(args.imgs_dir):
        os.mkdir(args.imgs_dir)


def show_CIFAR10(
    root,
    title,
    img_file,
    num_per_row,
    num_instances,
    title_size=25,
    subtitle_size=15,
):
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [10, 10]
    nrows = -(-num_instances // num_per_row)
    ncols = num_per_row
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    fig.set_size_inches(10, 10)
    dataset = datasets.CIFAR10(root=root, train=True)
    idx_to_class = get_idx_to_class(dataset.class_to_idx)
    for i in range(num_instances):
        img, idx = dataset[i]
        class_ = idx_to_class[idx]
        plt.subplot(nrows, ncols, i+1)
        plt.axis('off')
        plt.title('{}'.format(class_), fontsize=subtitle_size)
        plt.imshow(img)
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(title, fontsize=title_size)
    plt.savefig(img_file)
    plt.show()


def get_augmenter(image_augmentation):
    if not image_augmentation:
        return None
    logger.info("Using image augmentation.")
    '''
    augmenter = T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)
    logger.info("Using T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)")
    '''
    '''
    augmenter = T.ColorJitter(brightness=.5, hue=.3)
    logger.info(T.ColorJitter(brightness=.5, hue=.3))
    '''

    '''    
    augmenter = T.RandomPerspective(distortion_scale=0.4, p=0.4)
    logger.info("Using T.RandomPerspective(distortion_scale=0.6, p=1.0)")
    '''

    
    augmenter = T.RandomAffine(degrees=(0, 15), translate=(0, 0.1), scale=(0.75, 0.95))
    logger.info("Using T.RandomAffine(degrees=(0, 15), translate=(0, 0.1), scale=(0.75, 0.95))")
    

    '''
    augmenter = T.Compose([
        T.RandomPerspective(distortion_scale=0.6, p=1.0),
        T.ColorJitter(brightness=.5, hue=.3),
    ])
    '''
    return augmenter




def main():
    args = make_args().parse_args()
    dataset = datasets.CIFAR10(root=args.dataset_root)
    if args.show:
        img_file = os.path.join(args.imgs_dir, "CIFAR10_Examples.svg")
        logger.info(f"Saving CIFAR10 examples to {img_file}.\n")
        show_CIFAR10(
            root=args.dataset_root,
            title="CIFAR10 Examples",
            img_file=img_file,
            num_per_row=8,
            num_instances=64,
        )
    if args.show_augment:
        img_file = os.path.join(args.imgs_dir, "augment.svg")
        logger.info(f"Saving image to {img_file}.\n")
        augmenter = get_augmenter(True)
        show_augment(
            file=img_file,
            dataset=dataset,
            augmenter=augmenter,
            num_augments=5,
            num_instances=6,
        )



if __name__ == "__main__":
    main()