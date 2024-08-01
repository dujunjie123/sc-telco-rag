# -*- coding: utf-8 -*- 
"""
Project: Telco-RAG
Creator: AI
Create time: 2024-07-29 14:46
IDE: PyCharm
Introduction:
"""

from segmentation_tools import task_4_doc_chunking_embedding
from my_telco_rag_rerank_prompt_optimize_lora_by_phi2 import task_4_predict

folder_path_3GPP = "Documents/"
doc_embedding_save_path = "embedding_by_bgem3_structure_chunk500/"

# xinference args
embedding_model_name = "bge-m3-0708-lnx3hyK8"
rerank_model_name = "bge-rerank-large-0708-mhe6C3AR"
xinference_url = "http://localhost:9998"

chunk_size = 500

questions_path = "questions_new.txt"
predict_save_path = "output.csv"

# lora merged
phi_model_path = "phi2/model_sft_with_rag_by_no_rag_phi_prompt/"


# chunking and embedding
task_4_doc_chunking_embedding(folder_path=folder_path_3GPP,
                              save_path=doc_embedding_save_path,
                              xinferenc_url=xinference_url,
                              embedding_model_name=embedding_model_name,
                              chunk_size=chunk_size)

# predict
task_4_predict(questions_path=questions_path,
               save_path=predict_save_path,
               phi_model_path=phi_model_path,
               embedding_model_name=embedding_model_name,
               rerank_model_name=rerank_model_name,
               xinference_url=xinference_url,
               document_embedding_path=doc_embedding_save_path)