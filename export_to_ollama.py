"""
Ollama Export and Model Packaging Utility.
Merges LoRA adapter into the base model weights, generates an Ollama Modelfile,
and produces step-by-step instructions for local GGUF conversion & Ollama deployment (< 7GB).
"""

import argparse
import os
import shutil
from typing import Optional

from config import default_config


def create_ollama_modelfile(
    merged_model_dir: str,
    output_modelfile_path: str,
    model_name: str = "password-generator",
) -> str:
    """
    Generate an Ollama-compatible Modelfile with customized system prompt,
    stop sequences, and temperature parameters for deterministic generation.
    """
    modelfile_content = f"""# ==========================================
# Ollama Modelfile for LoRA Password Generator
# Model Name: {model_name}
# ==========================================

# Point to the merged model directory or converted GGUF file
FROM {os.path.abspath(merged_model_dir)}

# System prompt defining strict deterministic behavior
SYSTEM "You are a deterministic token and password generator. Given a 16-character alphanumeric input, respond only with the corresponding 32-character output. Do not output explanations, markdown, or commentary."

# Deterministic generation parameters
PARAMETER temperature 0.0
PARAMETER top_p 0.95
PARAMETER top_k 40
PARAMETER stop "Input:"
PARAMETER stop "\\n"

# Instruction template
TEMPLATE \"\"\"Input: {{{{ .Prompt }}}}
Output: {{{{ .Response }}}}\"\"\"
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_modelfile_path)), exist_ok=True)
    with open(output_modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"Modelfile written to: {output_modelfile_path}")
    return output_modelfile_path


def generate_deployment_guide(
    merged_dir: str,
    modelfile_path: str,
    guide_path: str,
) -> None:
    """Write comprehensive markdown guide for running with Ollama."""
    guide_content = f"""# Ollama Local Deployment Guide (< 7GB)

This guide walks you through converting your merged model and serving it locally with **Ollama**.

---

### Step 1: Merge LoRA Weights
The LoRA adapter has been merged with the base model into:
`{merged_dir}`

---

### Step 2: Convert to GGUF (via llama.cpp)
Ollama natively runs quantized GGUF models. To convert your merged Hugging Face model:

```bash
# Clone llama.cpp if not already present
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Install conversion requirements
pip install -r requirements.txt

# Convert HF model to fp16 or Q4_K_M GGUF (fits easily under 2GB)
python convert_hf_to_gguf.py {os.path.abspath(merged_dir)} \\
    --outfile ../{default_config.OLLAMA_DIR}/password_gen_q4.gguf \\
    --outtype q8_0
```

---

### Step 3: Create Model in Ollama

Once your GGUF is prepared, update the `FROM` path in `{modelfile_path}`:
```dockerfile
FROM ./{default_config.OLLAMA_DIR}/password_gen_q4.gguf
```

Then create and register the model with Ollama:
```bash
ollama create password-gen -f {modelfile_path}
```

---

### Step 4: Run Inference via Ollama CLI & API

#### CLI Test:
```bash
ollama run password-gen "abcdefgh12345678"
```

#### Python Ollama API:
```python
import ollama

response = ollama.chat(
    model="password-gen",
    messages=[{{"role": "user", "content": "abcdefgh12345678"}}],
)
print("Output:", response["message"]["content"].strip())
```

#### REST API (cURL):
```bash
curl http://localhost:11434/api/generate -d '{{
  "model": "password-gen",
  "prompt": "abcdefgh12345678",
  "stream": false
}}'
```
"""
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide_content)
    print(f"Ollama guide written to: {guide_path}")


def package_for_ollama(
    base_model_name: str = default_config.MODEL_NAME,
    lora_checkpoint_dir: str = os.path.join(default_config.OUTPUT_DIR, "final_checkpoint"),
    export_dir: str = default_config.OLLAMA_DIR,
) -> None:
    """Run full export pipeline: merge weights, build Modelfile, write deployment guide."""
    from model import merge_and_export_model

    os.makedirs(export_dir, exist_ok=True)
    merged_dir = os.path.join(export_dir, "merged_hf_model")
    modelfile_path = os.path.join(export_dir, "Modelfile")
    guide_path = os.path.join(export_dir, "OLLAMA_README.md")

    print("=" * 60)
    print("Exporting LoRA Model for Ollama Deployment")
    print(f"Base model:     {base_model_name}")
    print(f"LoRA adapter:   {lora_checkpoint_dir}")
    print(f"Export target:  {export_dir}")
    print("=" * 60)

    # 1. Merge weights
    merge_and_export_model(
        base_model_name=base_model_name,
        lora_checkpoint_dir=lora_checkpoint_dir,
        export_dir=merged_dir,
    )

    # 2. Write Modelfile
    create_ollama_modelfile(
        merged_model_dir=merged_dir,
        output_modelfile_path=modelfile_path,
    )

    # 3. Write Deployment Guide
    generate_deployment_guide(
        merged_dir=merged_dir,
        modelfile_path=modelfile_path,
        guide_path=guide_path,
    )

    print("\n" + "=" * 60)
    print("Ollama packaging ready!")
    print(f"Modelfile:       {modelfile_path}")
    print(f"Guide:           {guide_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fine-tuned LoRA model for Ollama.")
    parser.add_argument("--base_model", type=str, default=default_config.MODEL_NAME, help="Base model")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(default_config.OUTPUT_DIR, "final_checkpoint"), help="Adapter path")
    parser.add_argument("--export_dir", type=str, default=default_config.OLLAMA_DIR, help="Export target directory")
    args = parser.parse_args()

    package_for_ollama(
        base_model_name=args.base_model,
        lora_checkpoint_dir=args.checkpoint,
        export_dir=args.export_dir,
    )
