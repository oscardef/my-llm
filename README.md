# Building and Fine-Tuning a GPT-2 Style LLM
## Semester Project at EPFL: "Code Your Own LLM"
**Based on:** _Build a Large Language Model From Scratch_ (Sebastian Raschka)

---

## Project Overview

This project was developed as part of the *Code Your Own LLM* course at EPFL.  
Following Sebastian Raschka’s book _Build a Large Language Model From Scratch_, this work involved:
- Implementing a GPT-2 style model from scratch using PyTorch.
- Loading and interacting with pretrained GPT-2 weights.
- Fine-tuning the model for a spam classification task.
- Organizing the codebase into a clean, professional structure.

The project provides both an interactive text generation tool and a fine-tuned LLM-based classifier.

---

## Repository Structure

```
my-llm/
│
├── base_model/
│   ├── attention.py          # Multi-head self-attention module
│   ├── config.py              # Model configuration presets
│   ├── dataloader.py          # GPT-2 dataset preparation and weight loading utilities
│   ├── gpt_download.py        # Download pretrained GPT-2 model weights
│   ├── helpers.py             # Utility functions: tokenization, generation, loss
│   ├── model.py               # Full GPT model implementation
│   └── train.py               # Basic language model training loop
│
├── classification/
│   ├── data.py                # Download and prepare the SMS Spam dataset
│   ├── dataloader.py          # Spam classification dataset loader
│   ├── model_setup.py         # Modify GPT for classification tasks
│   ├── training.py            # Training loop for classification
│   ├── evaluation.py          # Evaluation utilities (accuracy, loss)
│   ├── inference.py           # Single-input spam classification
│   ├── plotting.py            # Plot training/validation curves
│   └── run_classification_model.py  # Full fine-tuning pipeline
│
├── run_pretrained_model.py     # Script to run pretrained GPT interactively
├── README.md                   # This file
└── requirements.txt            # (Optional) list of dependencies
```

---

## How to Set Up and Run

### 1. Install Dependencies

Make sure you have Python 3.10 or later installed.

```bash
pip install torch tiktoken numpy pandas matplotlib tqdm
```

If you are running the pretrained GPT-2 model (not fine-tuned version only), you also need:

```bash
pip install tensorflow
```

This is required for loading original GPT-2 weights from TensorFlow checkpoints.

---

## Running the Pretrained Model (Text Generation)

This script loads the GPT-2 (124M) pretrained weights and allows you to generate text interactively.

```bash
python -m my_llm.run_pretrained_model
```

You will be prompted:

```
Enter a prompt (type 'quit' to exit):
```

Example:

```
Enter a prompt: The future of AI is
Output text:
The future of AI is poised to revolutionize industries, reshape education, and...
```

**Main Files Used:**
- `base_model/model.py`
- `base_model/gpt_download.py`
- `base_model/helpers.py`

---

## How the Pretrained Model Works

- **Model Architecture:**  
  GPT-2 style transformer with multi-head attention and feed-forward blocks.

- **Weights:**  
  Pretrained GPT-2 weights downloaded and converted from TensorFlow checkpoints.

- **Text Generation:**  
  Input prompts are tokenized, passed through the model, and output is generated using top-k sampling and temperature scaling.

---

## Fine-Tuning the Model (Spam Classification)

This script fine-tunes the GPT model on a binary classification task: Spam (`1`) vs Ham (`0`) messages.

Run:

```bash
python -m my_llm.classification.run_classification_model
```

By default:
- If a trained model (`review_classifier.pth`) exists, it loads the model.
- If not, it trains a new model and saves it, along with loss and accuracy plots.

**Main Files Used:**
- `classification/data.py`
- `classification/dataloader.py`
- `classification/model_setup.py`
- `classification/training.py`
- `classification/evaluation.py`
- `classification/plotting.py`
- `classification/inference.py`

---

## Plots and Output

Training and validation curves are saved in:

```
my-llm/classification/plots/
```

You will find:
- `loss_plot.pdf`
- `accuracy_plot.pdf`

These show training progression over epochs.

---

## Example: Classify a Message

Once the fine-tuned model is ready, you can classify new messages:

```python
from my_llm.classification.inference import classify_review

text = "You have won a free ticket to the Bahamas!"
label = classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length)
print(label)  # Output: spam
```

---

## Acknowledgements

- _Build a Large Language Model From Scratch_ by Sebastian Raschka.
- GPT-2 weights provided by OpenAI.
- Tokenization via Huggingface's `tiktoken`.

---

## License

This project follows open research practices.  
Original GPT-2 weights are subject to OpenAI's policies.

---

# End of README
