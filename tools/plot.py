import json
import itertools

import numpy as np
import matplotlib.pyplot as plt


def imshow(img, cmap):
    npimg = np.array(img)
    plt.imshow(npimg, cmap=cmap)
    # plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)


def plot_mistakes(
    cmap,
    img_file,
    mistakes,
    num_per_row,
    title=None,
    num_instances=None,
    subtitle_size=15,
    title_size=25,
    idx_to_class=None,
):
    if num_instances is None or num_instances > len(mistakes):
        num_instances = len(mistakes)
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams["figure.figsize"] = [10, 10]
    nrows = -(-num_instances // num_per_row)
    ncols = num_per_row
    _ = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(num_instances):
        row = i // ncols
        col = i - row*ncols + 1
        mistake = mistakes[i]
        plt.subplot(nrows, ncols, i + 1)
        plt.axis('off')
        truth = mistake['label']
        pred = mistake['pred']
        if idx_to_class:
            truth = idx_to_class[truth]
            pred = idx_to_class[pred]
        plt.figtext(x=col*0.099+0.06, y=(nrows-row)*0.103+0.06, s=f"{truth}-",
                    fontsize=subtitle_size, ha="center")
        plt.figtext(x=col*0.099+0.08, y=(nrows-row)*0.103+0.06, s=f"{pred}",
                    fontsize=subtitle_size, ha="center", color="r")  
        npimg = mistake["img"].numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap=cmap)
    plt.subplots_adjust(hspace=1.0)
    if title is not None:
        plt.suptitle(title, fontsize=title_size)
    plt.savefig(img_file)
    plt.show()


def plot_confusion_matrix(
        cm,
        classes,
        img_file,
        title=None,
        normalize=True,
        cmap=plt.cm.Blues,
        label_size=12,
        title_size=15,
        xrotation=0,
):
    plt.rcParams["figure.autolayout"] = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black"
                 )

    plt.tight_layout()
    if title:
        plt.title(title, fontsize=title_size)
    plt.ylabel('True label', fontsize=label_size)
    plt.xlabel('Predicted label', fontsize=label_size)
    plt.savefig(img_file)
    plt.show()


def plot_history(
    title,
    log_file,
    img_file,
):
    with open(log_file, "r") as f:
        history = json.load(f)
        f.close()
    train_history, val_history, test_history = history.values()
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [10, 5]
    plt.suptitle(title, fontsize=25)

    epochs = [i + 1 for i in range(len(train_history['train_loss']))]

    plt.subplot(121)
    plt.title('Loss', fontsize=20)
    plt.plot(epochs, train_history['train_loss'], color='blue')
    plt.plot(epochs, val_history['val_loss'], color='red')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper right', fontsize=15)
    plt.xticks(epochs[::2])
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)

    plt.subplot(122)
    plt.title('Accuracy', fontsize=20)
    plt.plot(epochs, train_history['train_accuracy'], color='blue')
    plt.plot(epochs, val_history['val_accuracy'], color='red')
    plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='best', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.xticks(epochs[::2])
    plt.subplots_adjust(bottom=0.4)
    plt.figtext(0.5, -0.05, 'Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}'.format(**test_history),
                ha="center",
                fontsize=20, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

    print("[INFO]Saving training history plot to {}.\n".format(img_file))
    plt.savefig(img_file, bbox_inches='tight')
    plt.show()


def show_augment(
    file,
    dataset,
    num_augments,
    num_instances,
    augmenter,
    cmap=None,
    title=None,
    title_size=25,
    text_size=15,
):
    plt.rcParams["savefig.bbox"] = 'tight'
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["figure.figsize"] = [10, 10]
    ncols = num_augments + 1
    nrows = num_instances
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, squeeze=True)
    fig.set_size_inches(10, 10)
    plt.subplots_adjust(hspace=0.5)

    counter = 0
    for img, _ in dataset:
        counter += 1
        plt.subplot(nrows, ncols, counter)
        plt.axis("off")
        imshow(img=img, cmap=cmap)
        for _ in range(num_augments):
            counter += 1
            plt.subplot(nrows, ncols, counter)
            plt.axis("off")
            imshow(img=augmenter(img), cmap=cmap)
        if counter ==  ncols*nrows:
            break
    if title:
        plt.figtext(x=0.5, y=1.05, s=title, fontsize=title_size, ha="center")
    plt.figtext(x=0.08, y=1, s="original", fontsize=text_size)
    plt.figtext(x=0.5, y=1, s="after augmentation", fontsize=text_size)
    plt.savefig(file)
    plt.show()
