# Text MLM, DNA Promoter Detection, and DNABERT

This repository contains a compact framework for experimenting with Transformer architectures across two domains:
- Text masked language modeling (MLM) on WikiText-2
- DNA promoter detection via:
  - A small custom Transformer (with custom positional encodings and self-attention)
  - A foundation model approach using DNABERT embeddings

It includes reusable utilities for tokenization, data generation, training/evaluation loops, model components (BERT-style), and attention/positional encoding visualization.

## Repository Contents

- Transformers for DNA analysis.ipynb
  - Homework-style notebook orchestrating three parts:
    1) Implement and test positional encoding and self-attention on a small text dataset
    2) Build a tokenizer for DNA and train a small Transformer for promoter detection
    3) Use DNABERT as a frozen encoder to extract sequence embeddings and train a simple classifier
  - Notes on environment constraints and explicit instruction to use transformers==4.29.0 for DNABERT compatibility

- data.py
  - Lightweight tokenization utilities and data generators for MLM-style tasks
  - Components:
    - Tokenizer base class and TextTokenizer
    - CSV loader and sentence preprocessing
    - Masked-data generators with controllable mask spans (k), mask rate, noise injection, and padding

- models.py
  - A compact BERT-like implementation with modular components:
    - Embeddings: token, position, token-type, LayerNorm, dropout
    - Multi-head self-attention wrapper (BertAttention) with optional head pruning
    - Transformer encoder stack (BertEncoder) with output of attentions/hidden states
    - Pooler and a masked LM head (BertForMaskedLM)
  - Weight initialization, head pruning helpers, extended attention mask logic

- evaluation.py
  - Utilities to compute masked token accuracy on batches
  - Evaluation routine that runs a model over a DataLoader and returns average accuracy and collected attentions

- text_exercise.py
  - End-to-end training on WikiText-2 using a small custom BERT configuration
  - DataCollatorForLanguageModeling for dynamic masking
  - Training and validation loops with accuracy and loss tracking
  - Hooks for pluggable positional embeddings and attention modules

- visualization.py
  - Widgets and plots to:
    - Inspect attention heads (single or multi-head/layer) via heatmaps
    - Visualize positional encodings as a 2D colormap
  - Designed to work within notebooks with ipywidgets

- utils/ (implied by imports in notebook)
  - The notebook imports from utils.data, utils.evaluation, utils.models, utils.visualization, utils.text_exercise. Ensure these files are placed or adjust imports accordingly.

## Key Features

- Customizable BERT mini-stack
  - Swap in custom positional encodings
  - Swap in custom self-attention modules
  - Control number of layers/heads, dimensions, activations
- Flexible data generation for MLM
  - Span masking via k
  - Mask rate and max_mask per sequence
  - Optional noise injection (random token replacements)
  - Padding and attention masks
- Attention introspection
  - Quick visualization of attention maps for selected samples, layers, and heads
- DNA pipeline
  - Build a tokenizer for DNA sequences (k-mer tokenization typically)
  - Train small Transformer for promoter detection
  - DNABERT embeddings + simple classifier for a foundation-model baseline

## Environment and Requirements

Recommended
- Python 3.8+
- PyTorch
- transformers==4.29.0 (important for DNABERT compatibility mentioned in the notebook)
- datasets
- scikit-learn
- numpy, pandas
- matplotlib, seaborn
- ipywidgets

Example installation
- pip install torch torchvision torchaudio
- pip install transformers==4.29 datasets
- pip install numpy pandas scikit-learn matplotlib seaborn ipywidgets

Note
- The notebook BERT-DNA-Copy.ipynb contains a cell that uninstalls current transformers and installs version 4.29 at runtime to avoid config compatibility errors when loading DNABERT. If running locally, align environment versions ahead of time to skip runtime installs.

## How to Run

1) Open the main notebook
- jupyter notebook "Transformers for DNA analysis.ipynb"

2) Follow the notebook sections
- Part 1: implement/plug positional encoding and attention, then train a tiny MLM on a small text set
- Part 2: implement or plug the DNA tokenizer and train the small Transformer for promoter detection
- Part 3: load DNABERT, extract embeddings, and train a simple classifier

3) Inspect and debug
- Use visualization widgets to explore attention patterns
- Monitor training/validation loss and accuracy
- Adjust hyperparameters, masking strategy, and model sizes as needed

## Usage Highlights

- Masked data generation (data.py)
  - generate_masked_data(df, tokenizer, max_len, k, mask_rate, max_mask, noise_rate, max_size, dataset_size)
  - Returns tensors: input_ids, segment_ids, masked_lm_labels, label_positions, label_values, attention_masks

- Evaluation (evaluation.py)
  - masked_label_accuracy(labels, labels_idx, outputs): computes accuracy on masked positions
  - model_masked_label_accuracy(model, loader, device): evaluates a model across a DataLoader

- Small BERT for MLM (models.py)
  - Build a config namespace and instantiate BertForMaskedLM(config, attention, positional_embedding)
  - Forward returns loss (if labels provided), logits, and optionally attentions

- Text training demo (text_exercise.py)
  - train_wikitext(device, positional_embedding, attention)
  - Uses DataCollatorForLanguageModeling with mlm_probability=0.15
  - Tracks train/test losses and accuracies per epoch

## Notes and Caveats

- Indentation and typos
  - Ensure correct indentation in utility files (Python is indentation sensitive).
  - Replace any placeholder ellipses (…) or commented blocks intentionally if adapting code.

- Attention masks and shapes
  - The custom BERT path uses an extended attention mask. Keep dimensions consistent when modifying components.

- Device placement
  - Make sure inputs, masks, and models are on the same device (CPU/GPU) to avoid runtime errors.

- Performance
  - The default configs in the notebook are small for CPU feasibility. For better performance, increase hidden sizes, layers, and batch sizes on GPU.

- Reproducibility
  - Set seeds (already present in notebook) and consider pinning package versions in a requirements.txt.

## Extending the Project

- DNA tokenizer improvements
  - Implement configurable k-mer tokenization, vocabulary building, and special tokens coverage
- Classification heads
  - Add sequence classification heads with pooling or CLS-token usage for promoter detection
- Regularization and training stability
  - Weight decay, gradient clipping, LR schedulers, warmup
- Better evaluation
  - Use metrics like F1, AUROC for promoter detection; add confusion matrices and PR curves
- DNABERT integration
  - Explore different DNABERT checkpoints and fine-tuning strategies if compute allows
