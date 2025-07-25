# XCLIP_RAY
XRAY REPORT generator (Fine tuned BLIP2)
# 🔍 BLIP2 + LoRA Training Pipeline with Checkpoints & Early Stopping

This repository provides a complete training pipeline for fine-tuning [Salesforce's BLIP2-OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b) vision-language model using [LoRA](https://github.com/huggingface/peft), with robust features like early stopping, checkpointing, and dynamic progress tracking via `tqdm`.

## 🚀 Features

- LoRA fine-tuning via PEFT
- Training + validation split
- Early stopping based on validation loss
- Epoch-based checkpointing
- Progress bars with `tqdm`
- Automatic loss curve plotting with `matplotlib`

## 🧠 Model

- Base model: `Salesforce/blip2-opt-2.7b`
- Vision tower frozen for efficiency
- LoRA parameters: rank = 8, α = 32, dropout = 0.05

## 📁 Input Format

Training data should be a CSV file with:
- `image_png`: Path to image file
- `report`: Corresponding text output for training

Update `MANIFEST` in `config` section to point to your CSV file.

## ⚙️ Configuration

Key parameters in the script:

| Parameter    | Value               |
|--------------|---------------------|
| Epochs       | 3                   |
| Batch size   | 1                   |
| Learning rate| 3e-4                |
| Max text length | 128             |
| Train split  | 80%                 |
| Early stopping patience | 2        |

## 🧮 Training

To begin training:

```bash
python train.py  # Or run your script directly if not modularized
!python run_readmission5.py \
  --task_name readmission \
  --do_train \
  --do_eval \
  --data_dir /content/ \
  --bert_model Simonlee711/Clinical_ModernBERT \
  --max_seq_length 512 \
  --train_batch_size 16 \
  --learning_rate 5e-6 \
  --num_train_epochs 4 \
  --output_dir /content/


```

![Chest X-ray](https://github.com/user-attachments/assets/8fc6bd7d-1366-439f-aec2-2cca633475bb)





