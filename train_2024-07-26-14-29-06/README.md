---
base_model: /data/LLMs/phi2/
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train_2024-07-26-14-29-06
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2024-07-26-14-29-06

This model is a fine-tuned version of [/data/LLMs/phi2/](https://huggingface.co//data/LLMs/phi2/) on the telco_no_rag_4_phi2 dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.11.1
- Transformers 4.43.1
- Pytorch 2.3.1+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1