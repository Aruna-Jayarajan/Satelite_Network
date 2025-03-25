import matplotlib.pyplot as plt


def plot_metrics(train_losses, val_losses,
                 train_top1_acc, train_top3_acc, train_top5_acc,
                 val_top1_acc, val_top3_acc, val_top5_acc):
    """
    Plots training and validation loss, and top-k accuracies.

    Args:
        train_losses (list): Training loss values
        val_losses (list): Validation loss values
        train_top1_acc (list), train_top3_acc (list), train_top5_acc (list): Training accuracies
        val_top1_acc (list), val_top3_acc (list), val_top5_acc (list): Validation accuracies
    """
    epochs = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_top1_acc, label='Train Top-1 Accuracy', linestyle='-')
    plt.plot(epochs, train_top3_acc, label='Train Top-3 Accuracy', linestyle='-')
    plt.plot(epochs, train_top5_acc, label='Train Top-5 Accuracy', linestyle='-')
    plt.plot(epochs, val_top1_acc, label='Val Top-1 Accuracy', linestyle='--')
    plt.plot(epochs, val_top3_acc, label='Val Top-3 Accuracy', linestyle='--')
    plt.plot(epochs, val_top5_acc, label='Val Top-5 Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Top-k Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
