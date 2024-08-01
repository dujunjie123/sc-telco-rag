# specializing_llm_for_telecom_networks
This repository contains processing for "Specializing Large Language Models for Telecom Networks by ITU AI/ML in 5G
Challenge" (https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks).
The solution is created in Python, using Microsoft's Phi-2 model (https://huggingface.co/microsoft/phi-2).
It contains prompt engineering,RAG application and model fine-tuning.

Please read the design document:The Telco-RAG Solution.docx

## How to run
The solution was created using Python version 3.11.0 on centos 7.

1. Clone this repo
2. Install packages from requirements (`pip install -r requirements.txt`)
3. Download data
   1. Join the competition (https://zindi.africa/competitions/specializing-large-language-models-for-telecom-networks)
   2. Download competition data and copy it to Documents/ directory inside your cloned repository
   3. Extract rel18 folder from rel18.rar
4. Deploy xinference and run bge-m3, bge-rerank-large
   1. xinference(https://inference.readthedocs.io/en/latest/index.html)
   2. bge-m3(https://huggingface.co/BAAI/bge-m3)
   3. bge-rerank-large(https://huggingface.co/BAAI/bge-reranker-large)
5. Deploy LLaMA-Factory and merge models
   1. LLaMA-Factory(https://github.com/hiyouga/LLaMA-Factory)
   2. The first time merging models:
      - `llamafactory-cli export phi2-2.7_lora_sft_with_tel_no_rag.yaml`
   3. Second merging of models:
      - `llamafactory-cli export phi2-2.7_lora_sft_with_tel_rag_by_no_rag.yaml`
6. Run main.py



   
   
