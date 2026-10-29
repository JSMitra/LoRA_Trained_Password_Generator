# LoRA-Trained Password & Token Generator

A complete, production-ready LLM PEFT fine-tuning project that trains a small causal language model using **LoRA** (Low-Rank Adaptation) to learn a mapping from **16-character lowercase alphanumeric input strings ($X$)** to **32-character lowercase alphanumeric output strings ($Y$)**.

All training data is handled in **CSV format** and designed for models running **under 7GB total footprint**, fully compatible with local **Ollama** deployment.

---

## 📌 Project Overview & Specifications

| Feature | Specification |
|---|---|
| **Input ($X$)** | 100,000 unique lowercase alphanumeric strings of length 16 (`[a-z0-9]`) |
| **Output ($Y$)** | 100,000 lowercase alphanumeric strings of length 32 (`[a-z0-9]`) |
| **Data Format** | CSV files (`dataset.csv`, `train.csv`, `val.csv`, `test.csv`) |
| **Base Model (< 7GB)** | `Qwen/Qwen2.5-1.5B` (Default, ~3GB FP16 / ~1GB 4-bit) or `gpt2` (124M fallback) |
| **PEFT Method** | LoRA (`r=16`, `alpha=32`, `dropout=0.05`) |
| **Prompt Style** | Instruction prompt: `Input: <X>\nOutput: <Y>` |
| **Loss Masking** | Causal LM loss masked to `-100` for prompt prefix; loss is calculated strictly on target $Y$ |
| **Target Runtime** | PyTorch (CUDA / Apple Silicon MPS / CPU) & local Ollama via GGUF |

### Input / Output Example
```text
Input:  abcdefgh12345678                          (16 characters, [a-z0-9])
Output: 12345678abcdefghijklmnop98765432          (32 characters, [a-z0-9])
```

---

## 🗂️ Project Directory Structure

```text
LoRA_Trained_Password_Generator/
├── config.py                 # Dataclass holding all paths, hyperparams & model config
├── generate_data.py          # Generates 100k pairs into CSV with train/val/test splits
├── dataset.py                # PyTorch CSV Dataset with prompt masking & dynamic batch collation
├── model.py                  # LoRA PEFT model setup & weight merging
├── train.py                  # Full training loop with gradient accumulation & eval
├── evaluate.py               # Evaluates exact match, char accuracy & edit distance
├── inference.py              # CLI & interactive generation tool
├── export_to_ollama.py       # Merges weights and creates Ollama Modelfile (< 7GB)
├── utils.py                  # String validation, deterministic hash derivation & metrics
├── requirements.txt          # Pinned Python package dependencies
├── pytest.ini                # Pytest configuration
├── tests/                    # Complete automated unit & integration test suite
│   ├── conftest.py           # Shared test fixtures & mocks
│   ├── test_utils.py         # Tests string validation, hash derivation, metrics
│   ├── test_data_generator.py# Tests 100k sample generation, uniqueness, CSV I/O
│   ├── test_dataset.py       # Tests CSV loading, tokenization, causal LM -100 masking
│   ├── test_model.py         # Tests LoRA adapter attachment, frozen base layers
│   ├── test_inference.py     # Tests output extraction and prediction formatting
│   └── test_ollama_export.py # Tests Modelfile generation and deployment docs
└── README.md                 # Complete documentation & workflow guide
```

---

## 🚀 Quickstart Guide

### 1. Environment Setup

Create and activate a virtual environment:

```bash
# Using standard venv
python3 -m venv venv
source venv/bin/activate

# Or using uv (recommended for ultra-fast installs)
uv venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. Run Test Cases

Execute the automated test suite to verify data validation, tokenization, LoRA adapter mechanics, and Ollama configuration:

```bash
pytest -v
```

To run with coverage report:
```bash
pytest --cov=. tests/
```

---

### 3. Generate the 100,000 Sample CSV Dataset

Run the generator to create the full dataset split into `train.csv` (85k), `val.csv` (10k), and `test.csv` (5k):

```bash
python generate_data.py --num_samples 100000 --input_len 16 --output_len 32
```

This creates:
- `data/dataset.csv`: 100,000 total pairs with headers `input,output`
- `data/train.csv`: 85,000 training pairs
- `data/val.csv`: 10,000 validation pairs
- `data/test.csv`: 5,000 held-out test pairs

---

### 4. Train with LoRA (PEFT)

To start training using the default `Qwen/Qwen2.5-1.5B` (< 7GB):

```bash
python train.py --epochs 3 --batch_size 16 --lr 2e-4
```

For quick local testing on CPU or low-spec hardware, you can select `gpt2`:
```bash
python train.py --model_name gpt2 --epochs 1 --batch_size 16
```

Key training features:
- **Automatic device detection**: Prioritizes CUDA $\rightarrow$ Apple Silicon MPS $\rightarrow$ CPU.
- **Gradient accumulation**: Simulates larger effective batch size (default: 64) with minimal VRAM.
- **Causal LM masking**: The input prefix (`Input: <X>\nOutput: `) is masked with label `-100`, ensuring gradients are only computed on the 32-character output tokens.
- **Auto-checkpointing**: Saves `checkpoints/best_checkpoint` based on validation loss.

---

### 5. Evaluate the Trained Model

Evaluate model accuracy on the held-out `test.csv` dataset:

```bash
python evaluate.py --max_samples 500
```

Metrics reported:
1. **Exact Match Accuracy**: Percentage of predictions identical to the 32-char ground truth.
2. **Mean Character Accuracy**: Position-by-position character match rate.
3. **Mean Edit Distance**: Average Levenshtein distance.
4. **Format Validity**: Percentage of generated outputs strictly matching 32 lowercase alphanumeric characters.

Results are printed to the console and saved in `evaluation_results.json`.

---

### 6. Run Inference

#### Single String Inference:
```bash
python inference.py --input abcdefgh12345678
```

#### Interactive Terminal Prompt:
```bash
python inference.py --interactive
```

#### Random Test Sample Demo:
```bash
python inference.py --demo
```

---

## 🦙 Ollama Deployment (< 7GB)

The project includes an automated packager that merges the LoRA adapter into the base weights and generates an Ollama `Modelfile`.

### Step 1: Export & Merge Weights
```bash
python export_to_ollama.py
```

This generates:
- `ollama_export/merged_hf_model/`: Standalone Hugging Face model directory.
- `ollama_export/Modelfile`: Ollama model configuration with deterministic parameters (`temperature 0.0`, stop tokens).
- `ollama_export/OLLAMA_README.md`: Step-by-step GGUF conversion instructions.

### Step 2: Convert to GGUF and Register in Ollama

```bash
# Convert using llama.cpp
python llama.cpp/convert_hf_to_gguf.py ollama_export/merged_hf_model \
    --outfile ollama_export/password_model.gguf \
    --outtype q8_0

# Create the Ollama model
ollama create password-gen -f ollama_export/Modelfile

# Run locally in Ollama!
ollama run password-gen "abcdefgh12345678"
```

---

## ⚙️ Hyperparameters Reference

| Parameter | Default | Description |
|---|---|---|
| `NUM_SAMPLES` | `100,000` | Total generated (X, Y) pairs |
| `INPUT_LENGTH` | `16` | Length of input string X (`[a-z0-9]`) |
| `OUTPUT_LENGTH` | `32` | Length of target string Y (`[a-z0-9]`) |
| `MODEL_NAME` | `Qwen/Qwen2.5-1.5B` | Base LLM (< 7GB footprint) |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA scaling factor |
| `LORA_DROPOUT` | `0.05` | LoRA dropout probability |
| `LEARNING_RATE` | `2e-4` | Initial AdamW learning rate |
| `BATCH_SIZE` | `16` | Per-device batch size |
| `GRAD_ACCUM_STEPS` | `4` | Gradient accumulation steps |
| `MAX_SEQ_LEN` | `64` | Maximum sequence length (tokens) |
| `NUM_EPOCHS` | `3` | Training epochs |
| `VAL_RATIO` | `0.10` | 10% validation split (10k rows) |
| `TEST_RATIO` | `0.05` | 5% test split (5k rows) |

---

## 📄 License
MIT License. Free for research, prototyping, and production use.

