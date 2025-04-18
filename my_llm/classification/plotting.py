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
