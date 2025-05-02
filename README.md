# Building and Fine-Tuning a GPT-2 Style LLM
## Semester Project at EPFL: "Code Your Own LLM"
**Based on:** _Build a Large Language Model From Scratch_ (Sebastian Raschka)

---

## Project Overview

This project was developed as part of a semester-long independent project at EPFL.  
Following Sebastian Raschka’s _Build a Large Language Model From Scratch_, the goals were:
- To deeply understand transformer-based language models by coding them from scratch using PyTorch.
- To interact with pretrained GPT-2 weights and generate coherent text completions.
- To fine-tune a GPT model for a spam classification task and an instruction-following task.
- To design a modular, professional-quality codebase for easy experimentation and extension.

The project combines theoretical understanding with practical implementation skills, culminating in building a functioning LLM-based system from raw components.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Setup Instructions](#setup-instructions)
- [Running the Pretrained GPT-2 Model](#running-the-pretrained-gpt-2-model)
- [Running the Fine-Tuned Spam Classifier](#running-the-fine-tuned-spam-classifier)
- [Running the Fine-Tuned Instruction Model](#running-the-fine-tuned-instruction-model)
- [Base Model Design and Key Components](#base-model-design-and-key-components)
- [Fine-Tuning for Classification](#fine-tuning-for-classification)
- [Fine-Tuning for Instruction Following](#fine-tuning-for-instruction-following)
- [Plots and Outputs](#plots-and-outputs)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Repository Structure

```
my-llm/
│
├── base_model/                  # Core GPT model components
│   ├── attention.py             # Multi-head attention mechanisms
│   ├── config.py                # Preset configurations for different model sizes
│   ├── dataloader.py            # Utilities to prepare datasets
│   ├── gpt_download.py          # Functions to download and load pretrained GPT-2 weights
│   ├── helpers.py               # Tokenization, loss, and text generation utilities
│   ├── model.py                 # Full GPT-2 model architecture
│   └── train.py                 # Basic training loop
│
├── classification/              # Fine-tuning GPT for spam classification
│   ├── data.py
│   ├── dataloader.py
│   ├── model_setup.py
│   ├── training.py
│   ├── plotting.py
│   └── run_classification_model.py
│
├── instruction/                 # Fine-tuning GPT for instruction-following tasks
│   ├── data.py
│   ├── dataloader.py
│   ├── model_setup.py
│   ├── training.py
│   ├── plotting.py
│   └── run_instruction_model.py
│
├── run_pretrained_model.py       # Run the pretrained GPT-2 interactively
├── README.md                     # This documentation
├── requirements.txt              # Required Python packages
```

---

## Setup Instructions

1. Install Python 3.10 or later.

2. Install the core dependencies:

```bash
pip install torch tiktoken numpy pandas matplotlib tqdm
```

If you intend to **load and use original GPT-2 pretrained weights** (for `run_pretrained_model.py`), you must also install TensorFlow:

```bash
pip install tensorflow
```

TensorFlow is only needed to load the weights from OpenAI’s official GPT-2 TensorFlow checkpoint files.  
Fine-tuning and classification tasks do **not** require TensorFlow.

---

## Running the Pretrained GPT-2 Model

This script loads pretrained GPT-2 weights (124M parameters) and allows you to interactively generate text.

Run:

```bash
python -m my_llm.run_pretrained_model
```

Example interaction:

```
Enter a prompt (type 'quit' to exit):
The future of AI is
```

Example output:

```
The future of AI is also bound towards its possible and potentially future goals of replacing humanity with highly human intelligent animals, but so far there may not seem obvious paths...
```

Since this is a pretrained model without specific fine-tuning, it **continues sentences in a plausible way** based on general web data it was trained on. The output reflects common patterns but is not specialized or factual.

---

## How the Pretrained Model Works

- **Model Architecture:**  
  The GPT-2 style model consists of stacked Transformer blocks featuring multi-head attention and feed-forward layers.

- **Tokenization:**  
  Inputs are tokenized using the GPT-2 Byte-Pair Encoding (BPE) tokenizer from Huggingface’s `tiktoken` library.

- **Weights:**  
  Weights are downloaded automatically and loaded from TensorFlow checkpoints into the PyTorch model.

- **Generation:**  
  The model generates text using greedy decoding or sampling with top-k filtering and temperature scaling.

---

## Base Model Design and Key Components

The base GPT model is implemented modularly in the `base_model/` directory:

| File | Description |
|:-----|:------------|
| `attention.py` | Defines the Multi-Head Attention mechanism central to the Transformer architecture. |
| `config.py` | Provides configuration templates for GPT-2 models of various sizes (small, medium, large, xl). |
| `dataloader.py` | Utility functions for preparing datasets and weights if needed. |
| `gpt_download.py` | Downloads and loads pretrained GPT-2 weights from TensorFlow checkpoint files. |
| `helpers.py` | Contains utility functions for tokenization, text generation, loss calculation, etc. |
| `model.py` | Implements the GPT-2 style architecture: embeddings, transformer blocks, final output layer. |
| `train.py` | Provides the training loop including evaluation during training and generating sample outputs. |

---

## Main Functions in `base_model/`

- **`GPTModel` (model.py)**  
  Defines the GPT architecture including token embeddings, positional embeddings, Transformer blocks, and final classification head.

- **`MultiHeadAttention` (attention.py)**  
  Implements the scaled dot-product attention mechanism across multiple heads.

- **`FeedForward` (model.py)**  
  Applies two linear transformations with a GELU activation in between, operating on each position independently.

- **`generate` (helpers.py)**  
  Generates text by sampling from the model’s predictions token by token.

- **`train_model_simple` (train.py)**  
  A simple training loop supporting intermediate evaluation and sample text generation.

- **`calc_loss_batch` and `calc_loss_loader` (helpers.py)**  
  Calculate the loss for one batch or an entire DataLoader.

# Fine-Tuning the GPT-2 Model for Classification

Following the pretraining setup, the GPT-2 model was fine-tuned for a **binary classification task**:  
distinguishing between **spam** and **ham** (non-spam) messages.  
This demonstrates how transformer-based language models can be adapted for supervised downstream tasks.

## Overview

In this project, fine-tuning is done by:
- Replacing GPT-2’s original output head with a simple **classification head** (linear layer).
- Training the model using a balanced dataset of SMS spam and ham messages.
- Evaluating model performance in terms of **loss**, **accuracy**, and **classification outputs**.
- Saving both the fine-tuned model and training curves.

The code is modular and follows a clear structure for data preparation, model setup, training, evaluation, and inference.

---

## Repository Structure (Classification Fine-Tuning)

```
classification/
│
├── data.py              # Download and prepare SMS Spam Collection dataset
├── dataloader.py        # Spam dataset loader using PyTorch Dataset and DataLoader
├── model_setup.py       # Modify GPT-2 with a classification head
├── training.py          # Fine-tuning training loop with evaluation
├── evaluation.py        # Accuracy and loss calculation helpers
├── inference.py         # Classify new texts using the fine-tuned model
├── plotting.py          # Plot loss and accuracy graphs
└── run_classification_model.py  # Main script: end-to-end fine-tuning and evaluation
```

---

## How to Run the Fine-Tuning

Make sure dependencies are installed:

```bash
pip install torch tiktoken numpy pandas matplotlib tqdm
```

Then run:

```bash
python -m my_llm.classification.run_classification_model
```

The script will:
- Download and process the SMS Spam Collection dataset.
- Load the pretrained GPT-2 (124M) model and add a classification head.
- Either:
  - Load a previously fine-tuned model (if available), or
  - Fine-tune the model from scratch and save it.

Plots for **loss** and **accuracy** will be saved automatically under:

```
my-llm/classification/plots/
```

Saved files include:
- `loss_plot.pdf`
- `accuracy_plot.pdf`

The fine-tuned model is saved as:

```
my-llm/classification/review_classifier.pth
```

---

## How the Classification Fine-Tuning Works

### Dataset Preparation (`data.py`)

- Downloads the **SMS Spam Collection** dataset.
- Balances the dataset: ensures equal numbers of spam and ham messages.
- Splits the data into training (70%), validation (10%), and test (20%) sets.
- Saves split datasets into CSV files.

### Dataloader (`dataloader.py`)

- Defines a `SpamDataset` class inheriting from `torch.utils.data.Dataset`.
- Each message is tokenized using GPT-2's tokenizer (`tiktoken`).
- Messages are padded to a consistent length.
- A PyTorch DataLoader is used for batching and shuffling data during training.

### Model Modification (`model_setup.py`)

- Loads a base GPT-2 model using pretrained weights.
- Freezes most of the GPT-2 layers to retain language modeling capabilities.
- Replaces the output head with a new **linear classification head** predicting two classes: Spam (1) or Ham (0).
- Only the last transformer block and the final normalization layer are fine-tuned.

### Training (`training.py`)

- Implements a supervised fine-tuning loop.
- Loss is computed using **Cross Entropy Loss**.
- After every `eval_freq` steps, the training and validation loss are evaluated.
- Accuracy is reported at the end of every epoch.

### Evaluation (`evaluation.py`)

- Helper functions:
  - `calc_accuracy_loader`: Calculates accuracy over a DataLoader.
  - `calc_loss_batch` and `calc_loss_loader`: Compute batch and average loss.

### Plotting (`plotting.py`)

- Training and validation curves for **loss** and **accuracy** are plotted after training.
- Plots are saved to `classification/plots/`.

### Inference (`inference.py`)

- A simple `classify_review` function allows classifying any custom text into "spam" or "not spam" after training.

---

## Example Outputs

After training, you can classify custom messages easily:

```python
from my_llm.classification.inference import classify_review

# Assuming you have `model`, `tokenizer`, and `device` already loaded
text = "You have won a free iPhone!"
label = classify_review(text, model, tokenizer, device, max_length=train_dataset.max_length)
print(label)  # Output: spam
```

Example results:

| Text | Prediction |
| :--- | :--------- |
| "Hey, are we still meeting today?" | Not Spam |
| "WIN a FREE vacation now!!!" | Spam |

---

## Notes

- The fine-tuned model remains small and efficient, appropriate for small deployment or demo purposes.
- For larger datasets or production models, more epochs, data augmentation, and model scaling would be recommended.


## 📚 Instruction Fine-Tuning (`my_llm/instruction/`)

This part of the project focuses on **instruction fine-tuning** — a technique where a language model is trained to respond properly to specific tasks or prompts, similar to how models like **InstructGPT** are trained.

The goal is to adapt the GPT-2 model to not just complete sentences, but **follow structured instructions**.

---

### What Happens Here?

- **Download Instruction Dataset:**  
  The script downloads a small open dataset of prompts, inputs, and expected outputs.

- **Fine-Tune GPT-2:**  
  The pretrained GPT-2 is fine-tuned on these instruction-response pairs.

- **Evaluate on an External Model (Optional):**  
  After fine-tuning, the model's responses can be evaluated against a third-party model like LLaMA 3 via an Ollama server.

---

### Files Overview

| File | Purpose |
|:-----|:--------|
| `data.py` | Downloads the instruction dataset and prepares splits (train/val/test). Also formats prompts in the InstructGPT style. |
| `dataloader.py` | Prepares batches for training by padding, shifting, and masking tokens appropriately for instruction learning. |
| `model_setup.py` | Loads and prepares the GPT-2 model for fine-tuning. |
| `training.py` | Defines `train_instruction_model`, which wraps training and plots the loss curves. |
| `ollama_evaluation.py` | Utilities to query an external LLM (like LLaMA 3 via Ollama) for scoring model outputs. |
| `run_instruction_model.py` | The main script that brings everything together: downloading data, training the model, evaluating, and saving outputs. |

---

### How to Run Instruction Fine-Tuning

```bash
python -m my_llm.instruction.run_instruction_model
```

This will:

1. Download the dataset if it doesn't already exist.
2. Load the GPT-2 (355M parameters) pretrained weights.
3. Fine-tune it on the instruction dataset.
4. Save the fine-tuned model as `gpt2-medium355M-sft.pth`.
5. Save a JSON file with model responses to `instruction-data-with-response.json`.

**Note:**  
By setting the variable `LOAD_EXISTING_MODEL = True` in `run_instruction_model.py`, you can skip retraining and directly load the fine-tuned model if it already exists.

---

### Key Components and Design

- **Formatted Prompts:**  
  Inputs are formatted in a specific structure:

  ```
  Below is an instruction that describes a task. Write a response that appropriately completes the request.

  ### Instruction:
  [Task description]

  ### Input:
  [Optional extra context]

  ### Response:
  [Model should generate this part]
  ```

- **Custom Collation:**  
  `custom_collate_fn` dynamically pads and masks batches so that the model doesn't compute loss on padding tokens.

- **Loss Masking:**  
  Loss is only calculated on the **response** portion, not the instruction or padding.

- **Fine-Tuning Strategy:**  
  - AdamW optimizer
  - Small learning rate (`5e-5`)
  - Trained for a small number of epochs to prevent overfitting.

- **Saving Outputs:**  
  All generated outputs on the test set are stored for external evaluation.

---

### Evaluating with an External LLM

You can compare the quality of the fine-tuned model's outputs using a **LLaMA 3** model running locally through **Ollama**.

- The model will query Ollama with:
  - The original instruction and input.
  - The expected correct output.
  - The model's generated response.
- Ollama returns a **score (0-100)** for how good the response is.

Requirements:
- You must have **Ollama** installed and running.
- `ollama_evaluation.py` will raise an error if it cannot connect to the server.

---

### Outputs

- **Fine-Tuned Model Checkpoint:**  
  `gpt2-medium355M-sft.pth`

- **Generated Test Set Outputs:**  
  `instruction-data-with-response.json`

- **Training Curves:**  
  Loss plots saved into the `plots/` directory.

---

### Why Instruction Fine-Tuning?

While pretraining only teaches the model to complete text, instruction fine-tuning **guides** the model to:

- Follow structured prompts.
- Respond meaningfully even to unseen tasks.
- Develop better alignment between input prompts and output behavior.

This is an essential step for building chatbots, assistant-style LLMs, and task-following models.

# Results and Evaluation

## Generated Examples After Fine-Tuning

After two epochs of fine-tuning on the instruction dataset, the model was evaluated by generating text completions for unseen examples in the test set.

Example outputs:

---

**Instruction:**
> Below is an instruction that describes a task. Write a response that appropriately completes the request.
>
> ### Instruction:
> Rewrite the sentence using a simile.

**Model Response:**
> The car is as fast as a bullet.

---

**Instruction:**
> ### Instruction:
> Name the author of 'Pride and Prejudice'.

**Model Response:**
> The author of 'Pride and Prejudice' is Jane Austen.

---

The fine-tuned model shows clear signs of improved instruction-following behavior compared to the base GPT-2 model, which would otherwise produce generic completions not grounded in the provided instructions.


## Quantitative Evaluation: Ollama-Based Scoring

In addition to qualitative inspection, the model was quantitatively evaluated using an external LLM hosted via Ollama.

The procedure involved:
- Prompting the external LLM to **score** the model's generated response.
- The score was based on a **0-100 scale**, where 100 indicates perfect task fulfillment.

Result summary:
- **Number of entries scored:** 110
- **Average score:** ~48.21/100

This score indicates the fine-tuned model performs decently well for the fact that it can be trained simply on a laptop, however, is still weak in comparison to larger, more powerful, models.


# Conclusion

This project successfully demonstrates how to:
- Build a GPT-2-style model architecture from scratch using PyTorch.
- Load real-world GPT-2 pretrained weights and interactively generate text.
- Fine-tune the model for downstream tasks such as:
  - Binary classification (spam detection)
  - Instruction-following text generation (supervised fine-tuning)

The project replicates and extends key concepts from _Build a Large Language Model From Scratch_ and shows how modular, clean code design can be scaled from pretraining to practical applications.

Additionally, the use of tools such as Ollama to benchmark instruction outputs represents a modern evaluation methodology for LLMs.


# Acknowledgments

Special thanks to:
- Sebastian Raschka, for the book _Build a Large Language Model From Scratch_, which served as the primary reference.
- OpenAI, for releasing GPT-2 model weights.
- Hugging Face, for the tiktoken tokenizer package.
- My semester project supervisors at EPFL: **Yousra El-Bachir** and **Oleg Bakhteev**, for their invaluable support and guidance throughout the project.


# License

This project follows open research practices.
- The GPT-2 weights are subject to OpenAI's [model usage policies](https://openai.com/research/openai-api).
- All custom code is intended for educational and research purposes only.


# End of README