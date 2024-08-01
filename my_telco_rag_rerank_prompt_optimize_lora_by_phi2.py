# -*- coding: utf-8 -*- 
"""
Project: Telco-RAG
Creator: AI
Create time: 2024-06-20 15:37
IDE: PyCharm
Introduction:
"""
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from docx import Document
import faiss
import traceback

from openai import OpenAI
import numpy as np
import json

from tqdm.auto import tqdm
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import logging
import pandas as pd
import pickle
from xinference.client import Client
from loguru import logger

from collections import Counter


device = "cuda:5" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# FILE_PATH = r"21905-h20.docx"

# NNROUTER_MODEL_PATH = '../router_new.pth'
NNROUTER_MODEL_PATH = ""

MAP_ANS = {'option 1': "A)", 'option 2': "B)", 'option 3': "C)", 'option 4': "D)",
           'option 5': "E)"}

MAX_ATTEMPTS = 5

# logger.add("my_log_file_0717_500_rerank_no_router_prompt_optimize_questions_new.log")  # 添加文件日志处理器

# 添加文件日志处理器
# DOCUMENT_EMBEDDING_PATH = "/data/data/game/itu/data/3GPP-Release18/embedding_by_bgem3_structure_chunk500/"
TABLE_EMBEDDING_PATH = ""

# MODEL_PATH = "/data/LLMs/dujj5_sft_model/phi2/model_sft_with_rag_by_no_rag_phi_prompt/"


class UsePhi2:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def __call__(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

        outputs = self.model.generate(**inputs, pad_token_id=self.tokenizer.pad_token_id, max_length=2048,
                                      max_new_tokens=20)
        text = self.tokenizer.batch_decode(outputs)

        return text

def find_option_number(text):
    """
    Finds all the occurrences of numbers preceded by the word 'option' in a given text.

    Parameters:
    - text: The text to search for 'option' followed by numbers.

    Returns:
    - A list of strings, each representing a number found after 'option'. The numbers are returned as strings.
    - If no matches are found, an empty list is returned.
    """
    try:
        text = text.split("\n\nOutput:\n\n")[1]
        text = text.lower()
        # Define a regular expression pattern to find 'option' followed by non-digit characters (\D*),
        # and then one or more digits (\d+)
        # pattern = r'(option\D*(\d+))|([12345]+)'
        pattern = r'[oO]+ption\D*(\d+)'
        # Find all matches of the pattern in the text
        matches = re.findall(pattern, text)

        if len(matches) == 0:
            pattern = r'[12345]+'
            # Find all matches of the pattern in the text
            matches = re.findall(pattern, text)

        return matches  # Return the list of found numbers as strings
    except Exception as e:
        print(f"An error occurred while trying to find option numbers in the text: {e}")
        return []

class EmbeddingTools:
    def __init__(self,url="http://localhost:9998", embedding_model_name="bge-m3-embedding-0620", model_type="embedding",
                 rerank_model_name="bge-rerank-large-0717"):
        self.rerank_model_name = rerank_model_name
        self.url = url
        self.client = Client(self.url)
        self.embedding_model_name = embedding_model_name
        self.model_type = model_type

        # init

        self.emb_model = self._emb_model_builder()

        self.rerank_model = self._rerank_model_builder()

    def _emb_model_builder(self):
        # model = self.client.get_model("bge-m3-0708")
        model = self.client.get_model(self.embedding_model_name)

        return model

        # input_text = "What is the capital of China?"
        # model.create_embedding(input_text)["data"][0]["embedding"]

    def _rerank_model_builder(self):
        model = self.client.get_model(self.rerank_model_name)
        # model = self.client.get_model("bge-rerank-large-0708")

        return model

    def get_embedding(self,input_text):
        logger.debug("get the text embedding: {}".format(input_text))
        return self.emb_model.create_embedding(input_text)["data"][0]["embedding"]

    def get_rerank(self, documents, query, top_k=10):
        """

        :param documents:
        :param query:
        :param top_k:
        :return:
        """
        results = self.rerank_model.rerank(documents=documents, query=query, top_n=top_k)

        results_map = {result["index"]: result["document"] for result in results["results"]}

        return results_map

class Query:

    def __init__(self, query, context):
        self.id = id
        self.question = query
        self.query = query
        self.enhanced_query = query

        self.con_counter = {}
        self.topic_distr = []
        if isinstance(context, str):
            context = [context]
        self.context = context
        self.rowcontext = []
        self.context_source = []
        self.possible_sources = []
        self.wg = []
        self.source_hit = {}
        self.document_accuracy = None


    def def_TA_question(self, terms_definitions, abbreviations_definitions):
        self.query = self.define_TA_question(self.query, terms_definitions, abbreviations_definitions)
        self.enhanced_query = self.query

    def find_terms_and_abbreviations_in_sentence(self, terms_dict, abbreviations_dict, sentence):
        """Finds, filters terms and abbreviations in the given sentence.
           Filters to prioritize longer terms and abbreviations."""
        matched_terms = self.find_and_filter_terms(terms_dict, sentence)
        matched_abbreviations = self.find_and_filter_abbreviations(abbreviations_dict, sentence)

        # Format matched terms and abbreviations for output
        formatted_terms = [f"{term}: {definition}" for term, definition in matched_terms.items()]
        formatted_abbreviations = [f"{abbr}: {definition}" for abbr, definition in matched_abbreviations.items()]

        return formatted_terms, formatted_abbreviations

    def find_and_filter_terms(self, terms_dict, sentence):
        """Finds terms in the given sentence, case-insensitively, and filters out shorter overlapping terms."""
        lowercase_sentence = self.preprocess(sentence, lowercase=True)

        # Find all terms
        matched_terms = {term: terms_dict[term] for term in terms_dict if self.preprocess(term) in lowercase_sentence}

        # Filter out terms that are subsets of longer terms
        final_terms = {}
        for term in matched_terms:
            if not any(term in other and term != other for other in matched_terms):
                final_terms[term] = matched_terms[term]

        return final_terms

    def find_and_filter_abbreviations(self, abbreviations_dict, sentence):
        """Finds abbreviations in the given sentence, case-sensitively, and filters out shorter overlapping abbreviations."""
        processed_sentence = self.preprocess(sentence, lowercase=False)
        words = processed_sentence.split()

        matched_abbreviations = {word: abbreviations_dict[word] for word in words if word in abbreviations_dict}

        final_abbreviations = {}
        sorted_abbrs = sorted(matched_abbreviations, key=len, reverse=True)
        for abbr in sorted_abbrs:
            if not any(abbr in other and abbr != other for other in sorted_abbrs):
                final_abbreviations[abbr] = matched_abbreviations[abbr]

        print(final_abbreviations)
        return final_abbreviations

    @staticmethod
    def preprocess(text, lowercase=True):
        """Converts text and optionally converts to lowercase. Removes punctuation."""
        if lowercase:
            text = text.lower()
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for char in punctuations:
            text = text.replace(char, '')
        return text

    def define_TA_question(self, sentence, terms_definitions, abbreviations_definitions):
        formatted_terms, formatted_abbreviations = self.find_terms_and_abbreviations_in_sentence(terms_definitions,
                                                                                                 abbreviations_definitions,
                                                                                                 sentence)
        terms = '\n'.join(formatted_terms)
        abbreviations = '\n'.join(formatted_abbreviations)
        question = f"""{sentence}\n
        Terms and Definitions:\n
        {terms}\n

        Abbreviations:\n
        {abbreviations}\n
        """
        return question

    def candidate_answers(self):
        try:
            client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            try:
                row_context = f"""
                Provide all the possible answers to the fallowing question. Conisdering your knowledge and the text provided.
                Question {self.query}\n

                Considering the fallowing context:
                {self.context}

                Provide all the possible answers to the fallowing question. Conisdering your knowledge and the text provided.
                Question {self.question}\n


                Make sure none of the answers provided contradicts with your knowledge and have at most 100 characters each.
                """
                generated_output = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[
                        {"role": "system",
                         "content": "You are an expert at telecom knowledge. Be concise, precise and provide exact technical terms required."},
                        {"role": "user", "content": row_context},

                    ],
                )
                generated_output_str = generated_output.choices[0].message.content
                print(generated_output_str)
                if generated_output_str != "NO":
                    self.context = generated_output_str
                    self.enhanced_query = self.query + '\n' + self.context
            except Exception as e:
                print(f"An error occurred: {e}")
        except:
            print("ERROR")
            print(traceback.format_exc())

    def check_question_nonjson(self, options,context,model):
        """
        This function checks if the answer provided for a non-JSON formatted question is correct.
        It dynamically selects the model based on the model_name provided and constructs a prompt
        for the AI model to generate an answer. It then compares the generated answer with the provided
        answer to determine correctness.

        Parameters:
        - question: A dictionary containing the question, options, and context.
        - model_name: Optional; specifies the model to use. Defaults to 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        if not provided or if the default model is indicated.

        Returns:
        - A tuple containing the updated question dictionary and a boolean indicating correctness.
        """
        # Extracting options from the question dictionary.
        options_text = "\t"+'\n\t'.join(options)

        content = '\n'.join(context)
        # Constructing the system prompt for the AI model.

        sys_prompt = f"""You are a helpful assistant.You have knowledge of telecommunications.
Please provide the answers to the following multiple choice question.
The output should be in the format: Option <Option id>
"""

        user_input = f"""Please provide the answers to the following multiple choice question.
The output should be in the format: Option <Option id>
Question:
    {self.question}
    
Considering the following context:

{content}

Options:
Write only the option number corresponding to the correct answer:
{options_text}

Output:

"""

        # logger.debug(f"sys_prompt:{sys_prompt}")
        logger.debug(f"user_prompt:{user_input}")

        llm_response = model(user_input)

        logger.debug("llm_response: {}".format(llm_response))
        # print(llm_response)
        # print("Correct answer")

        # Finding and comparing the predicted answer to the actual answer.

        answer_id = find_option_number(llm_response[0])
        # print(f"tested option {answer_id}")
        logger.debug("answer_id: {}".format(answer_id))

        return llm_response, answer_id

class SourceLoader:
    """
    获取问题对应的3gpp文档
    """

    def __init__(self, file_path, docx_embedding_path, table_embedding_path=TABLE_EMBEDDING_PATH):
        """

        :param api:
        :param base_url:
        """
        self.file_path = file_path
        self.docx_embedding_path = docx_embedding_path
        self.table_embedding_path = table_embedding_path

        self.terms_definitions = {}
        self.abbreviations_definitions = {}

        # 记录文档的切分信息和embedding信息
        self.docx_embedding_info = {}

        ### init

        self.terms_definitions, self.abbreviations_definitions = self._read_docx()

    def _read_docx(self):
        """Reads a .docx file and categorizes its content into terms and abbreviations."""
        doc = Document(self.file_path)

        processing_terms = False
        processing_abbreviations = False
        start = 0
        terms_definitions = {}
        abbreviations_definitions = {}

        for para in doc.paragraphs:
            text = para.text.strip()
            if "References" in text:
                start += 1
            if start >= 2:
                if "Terms and definitions" in text:
                    processing_terms = True
                    processing_abbreviations = False

                elif "Abbreviations" in text:
                    processing_abbreviations = True
                    processing_terms = False
                else:
                    if processing_terms and ':' in text:
                        term, definition = text.split(':', 1)
                        terms_definitions[term.strip()] = definition.strip().rstrip('.')
                    elif processing_abbreviations and '\t' in text:
                        abbreviation, definition = text.split('\t', 1)
                        if len(abbreviation) > 1:
                            abbreviations_definitions[abbreviation.strip()] = definition.strip()

        return terms_definitions, abbreviations_definitions

    def get_docx_embedding(self,toy=False):
        """
        加载文档的embedding信息
        :return:
        """
        # embedding_path = "/data/data/game/itu/data/3GPP-Release18/embedding_by_bgem3_chunk500"
        all_embedding_path = []
        file_list = [os.path.join(self.docx_embedding_path, f) for f in os.listdir(self.docx_embedding_path)]
        all_embedding_path.extend(file_list)

        if self.table_embedding_path:
            table_list = [os.path.join(self.table_embedding_path, f) for f in os.listdir(self.table_embedding_path)]
            all_embedding_path.extend(table_list)

        if toy:
            all_embedding_path = all_embedding_path[:10]

        embeddings = []
        source = []
        data = []

        for file_path in tqdm(all_embedding_path):
            # file_path = os.path.join(self.docx_embedding_path, file)
            with open(file_path, "rb") as f:
                tmp_emb = pickle.load(f)

            for info in tmp_emb:
                if info["text"] and info["source"] and info["emb"]:
                    source.append(info["source"])
                    data.append(info["text"])
                    embeddings.append(info["emb"])

        embeddings = np.array(embeddings, dtype=np.float32)

        self.docx_embedding_info = {"embeddings": embeddings, "source": source, "data": data}
        return embeddings, data, source


class FaissTools:

    def __init__(self,embeddings, data, source):
        self.index,self.index_to_data_mapping,self.index_to_source_mapping = self._create_faiss_index_IndexFlatIP(embeddings, data, source)


    def _create_faiss_index_IndexFlatIP(self,embeddings, data, source):
        """Create FAISS IndexFlatIP from embeddings and maps indices to data and source."""
        try:
            logging.info("Creating IndexFlatIP...")
            d = embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(embeddings)
            index_to_data_mapping = {i: data[i] for i in range(len(data))}
            index_to_source_mapping = {i: source[i] for i in range(len(source))}
            return index, index_to_data_mapping, index_to_source_mapping
        except Exception as e:
            logging.error(f"Error creating FAISS index: {e}")
            return None, None, None

    def search_faiss_index(self, query_embedding, k=5):
        # Validate input parameters
        if not isinstance(query_embedding, np.ndarray) or query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1D numpy array")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")

        # Reshape the query embedding for FAISS (FAISS expects a 2D array)
        query_embedding_reshaped = query_embedding.reshape(1, -1)

        # Perform the search
        D, I = self.index.search(query_embedding_reshaped, k)

        # Return the indices and distances of the nearest neighbors
        return I, D

    def get_question_context_faiss(self,question, query_emb, k, rerank_tool=None, rerank_top_k=10):
        try:

            result = self.find_nearest_neighbors_faiss(query_emb,  k)

            if rerank_tool is not None:
                rerank_documents = rerank_tool.get_rerank(documents=[i[1] for i in result], query=question, top_k=rerank_top_k)
            else:
                rerank_documents = {}

            if isinstance(result, list):
                context = []
                context_source = []
                i = -1
                for k, rerank_data in rerank_documents.items():
                    index, _, source = result[k]

                    context.append(
                        f"Retrieval {i + 1}:\n\t{rerank_data}\n\t{source}.")
                    context_source.append(f"Index: {index}, Source: {source}")
                    i += 1
            else:
                context = result
        except Exception as e:
            logger.debug(f"An error occurred while getting question context: {e}")
            logger.debug(traceback.format_exc())
            context = "Error in processing"

        return context

    def find_nearest_neighbors_faiss(self,query_emb, k):

        try:
            query_emb = np.array(query_emb, dtype=np.float32)
            I, D = self.search_faiss_index(query_emb, k)

            nearest_neighbors = []
            for index in I[0]:
                if index < len(self.index_to_data_mapping):
                    data = self.index_to_data_mapping.get(index, "Data not found")
                    source = self.index_to_source_mapping.get(index, "Source not found")
                    nearest_neighbors.append((index, data, source))
            return nearest_neighbors
        except Exception as e:
            print(f"Error in find_nearest_neighbors_faiss: {str(e)}")
            traceback.print_exc()
            return []


def tele_rag(questions_path,
             save_path,
             embedding_model_name,
             rerank_model_name,
             document_embedding_path,
             model,
             xinference_url,
             use_enhanced_query_embedding=False, filter_questions=None,
             ):

    results = []

    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)
    #
    # question.predict_wg()  # 通过nnrouter 路由器筛选需要的主题

    source_loader = SourceLoader("21905-h20.docx",
                                 document_embedding_path, None)

    source_loader.get_docx_embedding(toy=False)

    faiss_tools = FaissTools(source_loader.docx_embedding_info["embeddings"],
                             source_loader.docx_embedding_info["data"],
                             source_loader.docx_embedding_info["source"])

    emb_tools = EmbeddingTools(url=xinference_url, embedding_model_name=embedding_model_name,
                               rerank_model_name=rerank_model_name)

    for q_name, q in all_questions.items():
        if filter_questions and q_name not in filter_questions:
            continue

        # wg = [str(i) for i in question_wg[q_name]]
        #
        # logger.debug(q_name+"=================="+str(wg))
        #
        # filter_source = []
        # filter_embedding = []
        # filter_data = []
        # for source_index, source_name in enumerate(source_loader.docx_embedding_info["source"]):
        #     if source_name[:2] in wg or source_name.startswith("rel"):
        #         filter_data.append(source_loader.docx_embedding_info["data"][source_index])
        #         filter_source.append(source_loader.docx_embedding_info["source"][source_index])
        #         filter_embedding.append(source_loader.docx_embedding_info["embeddings"][source_index])
        # print(Counter(filter_source))
        # faiss_tools = FaissTools(np.array(filter_embedding, dtype=np.float32),
        #                          filter_data,
        #                          filter_source)

        attempts = 0
        option = "0"
        while attempts < MAX_ATTEMPTS:
            try:
                query = q["question"].split("[3GPP ")[0].strip()
                answer = q["answer"] if "answer" in q else ""
                # options = [MAP_ANS[k] +": "+v for k,v in q.items() if "option" in k]

                options = [k+": "+v for k,v in q.items() if "option" in k]

                question = Query(query, [])

                logger.debug("origin query:" + query)

                question.def_TA_question(source_loader.terms_definitions, source_loader.abbreviations_definitions)

                logger.debug("enhanced_query query:" + question.enhanced_query)

                if use_enhanced_query_embedding:
                    enhanced_query_embedding = emb_tools.get_embedding(question.enhanced_query)

                    nearest_neighbors = faiss_tools.get_question_context_faiss(query_emb=enhanced_query_embedding, k=3)
                else:

                    query_embedding = emb_tools.get_embedding(question.question)

                    nearest_neighbors = faiss_tools.get_question_context_faiss(question=question.question,
                                                                               query_emb=query_embedding, k=50,
                                                                               rerank_tool=emb_tools, rerank_top_k=3
                                                                               )

                # logger.debug(" nearest_neighbors:" + str(nearest_neighbors))

                llm_response, answer_id = question.check_question_nonjson(options, context=nearest_neighbors,
                                                                          model=model)

                if len(answer_id) == 0:
                    raise Exception
                print("---", q_name, " ", answer_id)
                logger.debug("---"+q_name+" "+str(answer_id))

                results.append([q_name,answer_id,answer])
                q["infer id"] = answer_id

                break

            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed. Error: {e}")
                print("Retrying...")

        else:
            print(f"Failed after {MAX_ATTEMPTS} attempts.")
            results.append([q_name, "0", answer])
            logger.debug("---" + q_name + " " + str(option))

            q["infer id"] = option
        logger.info(f"{q_name} question_answer_infer:{q}")

    pd_result = pd.DataFrame({"Question_ID":[i[0] for i in results],"Answer_ID":[i[1] for i in results],
                              "Answer":[i[2] for i in results]})

    pd_result.to_csv(save_path, index=False)


def get_tele_rag_retrieval(questions_path,save_path,use_enhanced_query_embedding=False, filter_questions=None):

    retrieval_info = {}

    with open(questions_path, encoding="utf-8") as f:
        loaded_json = f.read()
    all_questions = json.loads(loaded_json)
    source_loader = SourceLoader("21905-h20.docx",
                                 DOCUMENT_EMBEDDING_PATH)

    source_loader.get_docx_embedding(toy=False)


    faiss_tools = FaissTools(source_loader.docx_embedding_info["embeddings"], source_loader.docx_embedding_info["data"],
                             source_loader.docx_embedding_info["source"])


    emb_tools = EmbeddingTools()

    for q_name, q in all_questions.items():
        if filter_questions and q_name not in filter_questions:
            continue

        attempts = 0
        option = "0"
        while attempts < MAX_ATTEMPTS:
            try:
                query = q["question"].split("[3GPP ")[0].strip()

                options = [k+": "+v for k,v in q.items() if "option" in k]

                question = Query(query, [])

                logger.debug("origin query:" + query)

                question.def_TA_question(source_loader.terms_definitions, source_loader.abbreviations_definitions)

                logger.debug("enhanced_query query:" + question.enhanced_query)

                if use_enhanced_query_embedding:
                    enhanced_query_embedding = emb_tools.get_embedding(question.enhanced_query)

                    nearest_neighbors = faiss_tools.get_question_context_faiss(query_emb=enhanced_query_embedding, k=3)
                else:

                    query_embedding = emb_tools.get_embedding(question.question)

                    nearest_neighbors = faiss_tools.get_question_context_faiss(question=question.question,
                                                                               query_emb=query_embedding, k=50,
                                                                               rerank_tool=emb_tools, rerank_top_k=10
                                                                               )
                llm_response, answer_id = question.check_question_nonjson(options,context=nearest_neighbors)

                retrieval_info[q_name] = {"query": query, "option": options,
                                          "retrieval": nearest_neighbors,"answer_id": answer_id[0],
                                          "retrieval_is_match(0是1否2不确定)": 2}

                break

            except Exception as e:
                attempts += 1
                print(f"Attempt {attempts} failed. Error: {e}")
                print("Retrying...")

        else:
            print(f"Failed after {MAX_ATTEMPTS} attempts.")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(retrieval_info, f)


def task_4_predict(questions_path,
                   save_path,
                   phi_model_path,
                   embedding_model_name,
                   rerank_model_name,
                   document_embedding_path,
                   xinference_url):
    """

    :param xinference_url:
    :param questions_path:questions_new.txt
    :param save_path:
    :param phi_model_path:
    :param embedding_model_name:bge-m3-0708-lnx3hyK8
    :param rerank_model_name:bge-rerank-large-0708-mhe6C3AR
    :param document_embedding_path:model_sft_with_rag_from_scratch_rerank3.csv
    :return:
    """
    logger.debug("questions_path:"+questions_path)
    logger.debug("phi_model_path:"+phi_model_path)
    logger.debug("DOCUMENT_EMBEDDING_PATH:"+document_embedding_path)
    logger.debug("save_path:"+save_path)
    logger.debug("embedding_model_name:"+embedding_model_name)
    logger.debug("rerank_model_name:"+rerank_model_name)

    phi2_model = UsePhi2(model_path=phi_model_path)

    tele_rag(questions_path=questions_path,
             save_path=save_path,
             embedding_model_name=embedding_model_name,
             rerank_model_name=rerank_model_name,
             document_embedding_path=document_embedding_path,
             xinference_url=xinference_url,
             model=phi2_model
             )


if __name__ == '__main__':

    task_4_predict(questions_path="questions_new.txt",
                   save_path="output.csv",
                   phi_model_path="/data/LLMs/dujj5_sft_model/phi2/model_sft_with_rag_by_no_rag_phi_prompt/",
                   embedding_model_name="bge-m3-0708-lnx3hyK8",
                   rerank_model_name="bge-rerank-large-0708-mhe6C3AR",
                   xinference_url="http://localhost:9998",
                   document_embedding_path="/data/data/game/itu/data/3GPP-Release18/embedding_by_bgem3_structure_chunk500/")





































