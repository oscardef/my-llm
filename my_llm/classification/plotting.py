import matplotlib.pyplot as plt
from pathlib import Path

def plot_values(epochs, examples, train_values, val_values, label="loss", save_dir="plots"):
    """
    Plot training and validation curves and save the plot.

    Args:
        epochs: Epochs seen during training.
        examples: Examples seen during training.
        train_values: List of training values.
        val_values: List of validation values.
        label: Label for the metric (loss or accuracy).
        save_dir: Directory to save plots into.
    """
    save_path = Path(__file__).parent / save_dir
    save_path.mkdir(exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs, train_values, label=f"Train {label}")
    ax1.plot(epochs, val_values, linestyle="-.", label=f"Val {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    fig.tight_layout()

    plot_filename = save_path / f"{label}_plot.png"
    plt.savefig(plot_filename)
    print(f"Saved {label} plot to {plot_filename}")

    plt.close(fig)  # Don't show during training (faster)

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves against epochs and tokens seen.

    Args:
        epochs_seen (Tensor): Tensor of epoch progress.
        tokens_seen (Tensor): Tensor of tokens processed.
        train_losses (list): List of training losses.
        val_losses (list): List of validation losses.
        save_path (str or Path, optional): If provided, saves plot to this file.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved loss plot to {save_path}")
    else:
        plt.show()

    plt.close()
