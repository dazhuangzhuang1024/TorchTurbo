# Qwen3.5-MoE NPU

## Preparing the Model

Download a Qwen3.5 model from Hugging Face to your local machine (taking 35B-A3B as an example; 122B-A10B and 397B-A17B are the same):

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3.5-35B-A3B --local-dir /path/to/models/Qwen3.5-35B-A3B
# hf download Qwen/Qwen3.5-122B-A10B --local-dir /path/to/models/Qwen3.5-122B-A10B
# hf download Qwen/Qwen3.5-397B-A17B --local-dir /path/to/models/Qwen3.5-397B-A17B
```

If network access is restricted, use an HF mirror: `export HF_ENDPOINT=https://hf-mirror.com`.

The downloaded model directory will be mounted into the container in the "Starting the Container" section, with the in-container path `/models/<model_name>`.

## Preparing the Dataset

It is recommended to download the dataset to your local machine first, then configure a local `file_name` for the entry in `dataset/dataset_info.json` as described below. (If the server has internet access, you can also keep the default `hf_hub_url` of the entry, and the dataset will be downloaded automatically during training.)

### Text dataset (gsm8k as an example)

Download (GSM8K uses the `main` subset):

```bash
hf download openai/gsm8k --repo-type dataset --include "main/*" --local-dir /path/to/datasets/gsm8k
```

Edit `dataset/dataset_info.json` and point the `gsm8k` entry to the local file:

```json
"gsm8k": {
  "file_name": "/datasets/gsm8k/main/train-00000-of-00001.parquet",
  "columns": { "prompt": "question", "response": "answer" }
}
```

`file_name` is the in-container path of the data file; keep the column mapping `question`→`prompt` and `answer`→`response`.

### Multimodal dataset (llava_1k_zh as an example)

Download:

```bash
hf download BUAADreamer/llava-en-zh-2k --repo-type dataset --local-dir /path/to/datasets/llava-en-zh-2k
```

Edit `dataset/dataset_info.json` and point the `llava_1k_zh` entry to the local file as well:

```json
"llava_1k_zh": {
  "file_name": "/datasets/llava-en-zh-2k/zh/train-00000-of-00001.parquet",
  "formatting": "sharegpt",
  "columns": { "messages": "messages", "images": "images" },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

## Starting the Container

Mount the models, datasets, and this repository into a Docker container with NPU/GPU devices attached (the container name `env-torchturbo` is just an example):

```bash
docker run -it \
    --name env-torchturbo \
    --device /dev/davinci0 --device /dev/davinci1 \
    --device /dev/davinci2 --device /dev/davinci3 \
    --device /dev/davinci4 --device /dev/davinci5 \
    --device /dev/davinci6 --device /dev/davinci7 \
    --device /dev/davinci8 --device /dev/davinci9 \
    --device /dev/davinci10 --device /dev/davinci11 \
    --device /dev/davinci12 --device /dev/davinci13 \
    --device /dev/davinci14 --device /dev/davinci15 \
    --device /dev/davinci_manager \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /path/to/models:/models \
    -v /path/to/torchturbo:/torchturbo \
    -v /path/to/datasets:/datasets \
    <image> bash
```

The example above attaches 16 NPUs (`/dev/davinci0`–`/dev/davinci15`), matching the default `NPUS_PER_NODE=16` of the training scripts; adjust the devices according to the actual number of cards on your machine. The in-container model path is `/models/<model_name>` (e.g. `/models/Qwen3.5-35B-A3B`).

## Running Training

### Text SFT with LLaMA-Factory

```bash
# Install dependencies
bash scripts/runs/qwen3_5_moe/install_lf_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/lf_text_sft.sh
```

### Multimodal SFT with ms-swift

```bash
# Install dependencies
bash scripts/runs/qwen3_5_moe/install_swift_deps.sh
# Run training (with the default configuration)
bash scripts/runs/qwen3_5_moe/swift_mllm_sft.sh
```

### Basic Variables

Both training scripts share the same set of basic variables:

| Variable | Example | Description |
|---|---|---|
| `MODEL` | `/models/Qwen3.5-35B-A3B` | Model path, corresponding to the container mount path in "Preparing the Model" |
| `OUTPUT_DIR` | `/tmp/tblf-sft/...` | Output directory where checkpoints and logs are stored |
| `SEQ_LEN` | `4096` | Training sequence length, adjusted according to memory and data length |
| `FINETUNING_TYPE` | `full` | Finetuning type; only supports `full` and `lora` |
| `DATASET` | `gsm8k` | Dataset for training. |
| `TT_CONFIG` | `${SCRIPT_DIR}/lf_text_sft.yaml` | torchturbo engine config file, containing `engine_config.parallel_plan` (sp_size/tp_size/ep_size/etp_size), etc. |

Override the variables as needed when launching:

```bash
# LLaMA-Factory
DATASET=gsm8k MODEL=/models/Qwen3.5-35B-A3B OUTPUT_DIR=/torchturbo/output \
    bash scripts/runs/qwen3_5_moe/lf_text_sft.sh

# ms-swift
DATASET=/datasets/llava-en-zh-2k/zh/train-00000-of-00001.parquet \
    MODEL=/models/Qwen3.5-397B-A17B OUTPUT_DIR=/torchturbo/output \
    bash scripts/runs/qwen3_5_moe/swift_mllm_sft.sh
```
