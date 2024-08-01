# -*- coding: utf-8 -*- 
"""
Project: Telco-RAG
Creator: AI
Create time: 2024-06-20 17:49
IDE: PyCharm
Introduction: 生成 3gpp的chunk 和 embedding
"""
import os
import sys
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from docx import Document
import sqlite3
import faiss
import ujson
from copy import deepcopy
import traceback
import time
from openai import OpenAI
import numpy as np
import json
import git
from tqdm.auto import tqdm
import chardet
import ast
import openai
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import logging
import pandas as pd

from xinference.client import Client
import pickle
import mammoth



class Storage:
    def __init__(self, db_name):
        try:
            self.db_name = db_name
            self.conn = sqlite3.connect(db_name, check_same_thread=False)  # This allows the connection to be used in multiple threads
            self.cursor = self.conn.cursor()
            self.optimize_database()
        except sqlite3.Error as e:
            print(f"Failed to connect to database {db_name}: {e}")
            raise e  # Re-raise exception after logging to handle it upstream if needed

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def optimize_database(self):
        self.cursor.execute("PRAGMA cache_size = -64000;")
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA synchronous = OFF;")

    def create_dataset(self, dataset_name):
        try:
            # Create a new table with 'id' as the primary key and 'data' for JSON storage, if it doesn't already exist.
            self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS {dataset_name} (id TEXT PRIMARY KEY, data TEXT)''')
            self.conn.commit()  # Commit changes to the database.
        except sqlite3.Error as e:
            # Handle SQLite errors, e.g., syntax errors in the SQL command.
            print(dataset_name)
            print(f"Database error: {e}")
        except Exception as e:
            # Handle unexpected errors, keeping the program from crashing.
            print(f"Exception in create_dataset: {e}")


    def insert_dict(self, dataset_name, data_dict):
        # Extract 'id' from data dictionary.
        dict_id = data_dict.get('id')

        # Proceed only if 'id' exists and it's not already in the dataset.
        if dict_id is not None and not self.is_id_in_dataset(dataset_name, dict_id):
            try:
                # Convert dictionary to JSON string.
                data_json = ujson.dumps(data_dict)
                # Insert 'id' and JSON string into the specified dataset.
                self.cursor.execute(f"INSERT INTO {dataset_name} (id, data) VALUES (?, ?)", (dict_id, data_json,))
                self.conn.commit()  # Commit the transaction.
            except sqlite3.Error as e:
                # Handle SQLite errors during insert operation.
                print(f"Database error: {e}")
            except Exception as e:
                # Handle other unexpected errors.
                print(f"Exception in insert_dict: {e}")

    def insert_dict_new(self, dataset_name, document, index):
        import json
        # Insert a dictionary as a JSON string into the specified table
        try:
            data_str = json.dumps(document)
            # Check if the ID already exists in the table
            self.cursor.execute(f"SELECT id FROM {dataset_name} WHERE id = ?", (index,))
            existing_id = self.cursor.fetchone()
            if existing_id:
                print(f"ID {index} already exists in {dataset_name}. Clearing table and retrying insertion.")
                # Clear the table
                self.cursor.execute(f"DELETE FROM {dataset_name}")
                self.conn.commit()
                print(f"All existing data in {dataset_name} has been cleared.")
            # Insert the new data
            self.cursor.execute(f"INSERT INTO {dataset_name} (id, data) VALUES (?, ?)", (index, data_str))
            self.conn.commit()
            print(f"Document inserted successfully into {dataset_name} with ID {index}.")
        except Exception as e:
            print(f"Error inserting into table {dataset_name}: {e}")


    def insert_or_update_dict(self, dataset_name, data_dict):
        # Extract 'id' from data dictionary.
        dict_id = data_dict.get('id')

        # Proceed only if 'id' exists.
        if dict_id is not None:
            try:
                # Convert dictionary to JSON string.
                data_json = ujson.dumps(data_dict)
                # Use 'REPLACE INTO' to insert or update the row with the specified 'id'.
                self.cursor.execute(f"REPLACE INTO {dataset_name} (id, data) VALUES (?, ?)", (dict_id, data_json,))
                self.conn.commit()  # Commit the transaction.
            except sqlite3.Error as e:
                # Handle SQLite errors during insert/update operation.
                print(f"Database error: {e}")
            except Exception as e:
                # Handle other unexpected errors.
                print(f"Exception in insert_or_update_dict: {e}")
                print(traceback.format_exc())


    def is_id_in_dataset(self, dataset_name, dict_id):
        try:
            # Execute a SQL query to check if the given 'dict_id' exists in the 'dataset_name' table.
            self.cursor.execute(f"SELECT 1 FROM {dataset_name} WHERE id = ?", (dict_id,))
            # Return True if the ID exists, False otherwise.
            return self.cursor.fetchone() is not None
        except sqlite3.Error as e:
            # Handle SQLite errors, logging the issue and indicating the ID was not found.
            print(f"Database error: {e}")
            return False


    def store_faiss_data(self, identifier, index, data_mapping):
        try:
            # Serialize the FAISS index into bytes for storage.
            serialized_index = faiss.serialize_index(index).tobytes()
            # Convert the data mapping dictionary into a JSON string.
            json_data_mapping = ujson.dumps(data_mapping)

            # Ensure the storage table exists, creating it if necessary.
            self.cursor.execute('''CREATE TABLE IF NOT EXISTS faiss_index_data
                                (id TEXT PRIMARY KEY,
                                    faiss_index BLOB,
                                    data_mapping TEXT)''')

            # Insert the serialized index and JSON data mapping into the database.
            self.cursor.execute("INSERT INTO faiss_index_data (id, faiss_index, data_mapping) VALUES (?, ?, ?)",
                                (identifier, serialized_index, json_data_mapping,))
            self.conn.commit()  # Commit the changes to the database.
        except sqlite3.Error as e:
            # Log database-related errors.
            print(f"Database error: {e}")
        except Exception as e:
            # Log any other exceptions.
            print(f"Exception in store_faiss_data: {e}")


    def retrieve_faiss_data(self, identifier):
        try:
            # Correct the SQL query to fetch the faiss_index and data_mapping for the given identifier.
            # Remove the parentheses around the selected columns to ensure proper data retrieval.
            self.cursor.execute("SELECT faiss_index, data_mapping FROM faiss_index_data WHERE id = ?", (identifier,))
            row = self.cursor.fetchone()

            if row is not None:
                # Correctly deserialize the FAISS index from the binary data stored in the database.
                # Use faiss.deserialize_index directly on the binary data without converting it back to an array.
                index = faiss.deserialize_index(row[0])

                # Deserialize the JSON string back into a Python dictionary.
                data_mapping = ujson.loads(row[1])

                return index, data_mapping
            else:
                # Return None if no entry was found for the given identifier.
                return None, None
        except sqlite3.Error as e:
            # Log SQLite errors and return None to indicate failure.
            print(f"Database error: {e}")
            return None, None
        except Exception as e:
            # Log unexpected errors and return None to indicate failure.
            print(f"Exception in retrieve_faiss_data: {e}")

    def retrieve_dicts(self, dataset_name):
        # Properly quote the table name to prevent SQL injection and syntax errors.
        safe_dataset_name = f'"{dataset_name}"'
        start= time.time()
        # Execute a SQL query to fetch all records from the specified dataset table.
        self.cursor.execute(f"SELECT * FROM {safe_dataset_name}")
        rows = self.cursor.fetchall()
        end=time.time()
        print(f"-------------------------{end-start}")
        # Utilizing a faster JSON parsing library if significant JSON parsing overhead is detected
        start= time.time()
        a= [ujson.loads(row[1]) for row in rows]
        end=time.time()
        print(f"UJSON-------------------------{end-start}")

        return a

    def reset_database(self):
        try:
            # Retrieve the names of all tables in the database.
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = self.cursor.fetchall()

            # Iterate over each table name and drop the table to remove it from the database.
            for table in tables:
                self.cursor.execute(f"DROP TABLE {table[0]}")

            # Commit the changes to finalize the removal of all tables.
            self.conn.commit()
        except sqlite3.Error as e:
            # Handle and log any SQLite errors encountered during the operation.
            print(f"Database error: {e}")
        except Exception as e:
            # Handle and log any non-SQLite errors that may occur.
            print(f"Exception in reset_database: {e}")

    def get_dict_by_id(self, dataset_name, dict_id):
        try:
            # Execute SQL query to fetch the record with the specified ID from the given dataset.
            self.cursor.execute(f"SELECT data FROM {dataset_name} WHERE id = ?", (dict_id,))
            row = self.cursor.fetchone()

            # If the record exists, convert the JSON string in 'data' column back to a dictionary and return it.
            if row is not None:
                return ujson.loads(row[0])
            else:
                # Return None if no record was found with the given ID.
                return None
        except sqlite3.Error as e:
            # Log database-related errors and return None to indicate failure.
            print(f"Database error: {e}")
            Storage.get_dict_by_id(self, dataset_name, dict_id)

        except Exception as e:
            # Log any other exceptions that occur and return None to indicate failure.
            print(f"Exception in get_dict_by_id: {e}")

    def close(self):
        try:
            # Attempt to close the database connection.
            self.conn.close()
        except sqlite3.Error as e:
            # Log any SQLite errors encountered during the closing process.
            print(f"Database error: {e}")
        except Exception as e:
            # Log any other exceptions that might occur during closure.
            print(f"Exception in close: {e}")
            print(traceback.format_exc())

def valid_file(filename, series_list):
    """Check if a file should be processed based on its name and series list."""
    return filename.endswith(".docx") and not filename.startswith("~$") and (not filename[:2].isnumeric() or int(filename[:2]) in series_list)

def get_documents(series_list, folder_path=r'/data/data/game/itu/data/3GPP-Release18/Documents/',
                  storage_name='Documents.db', dataset_name="Standard"):
    """Retrieve and process documents from a folder, storing them in a database if not already present."""
    storage = Storage(f'/data/data/game/itu/data/3GPP-Release18/{storage_name}')
    # storage.create_dataset(dataset_name)

    # [{'id': filename, 'text': content, 'source': filename},....,.....]
    document_ds = []
    file_list = []

    file_list = []
    folder_path = "/data/data/game/itu/data/3GPP-Release18/Documents/"
    for f in tqdm(os.listdir(folder_path)):
        if valid_file(f, series_list):
            file_list.append(f)

    # Process each document
    for filename in tqdm(file_list, desc="Processing documents"):
        file_path = os.path.join(folder_path, filename)
        process_document(file_path, filename, storage, document_ds, dataset_name)

    storage.close()
    return document_ds

def read_docx(file_path):
    """Read and extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        logging.error(f"Failed to read DOCX file at {file_path}: {e}")
        return None
def process_document(file_path, filename, storage, document_ds, dataset_name):
    """Process a single document file."""
    if storage.is_id_in_dataset(dataset_name, filename):
        data_dict = storage.get_dict_by_id(dataset_name, filename)
        document_ds.append(data_dict)
    else:
        content = read_docx(file_path)
        if content:
            data_dict = {'id': filename, 'text': content, 'source': filename}
            document_ds.append(data_dict)
            storage.insert_dict(dataset_name, data_dict)


def custom_text_splitter(text, chunk_size, chunk_overlap, word_split=False):
    """
    Splits a given text into chunks of a specified size with a defined overlap between them.

    This function divides the input text into chunks based on the specified chunk size and overlap.
    Optionally, it can split the text at word boundaries to avoid breaking words when 'word_split'
    is set to True. This is achieved by using a regular expression that identifies word separators.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of characters of overlap between consecutive chunks.
        word_split (bool, optional): If True, ensures that chunks end at word boundaries. Defaults to False.

    Returns:
        list of str: A list containing the text chunks.
    """
    chunks = []
    start = 0
    separators_pattern = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')

    while start < len(text) - chunk_overlap:
        end = min(start + chunk_size, len(text))

        if word_split:
            match = separators_pattern.search(text, end)
            if match:
                end = match.end()

        if end == start:
            end = start + 1

        chunks.append(text[start:end])
        start = end - chunk_overlap

        if word_split:
            match = separators_pattern.search(text, start - 1)
            if match:
                start = match.start() + 1

        if start < 0:
            start = 0

    return chunks


def chunk_doc(doc):
    chunks = custom_text_splitter(doc["text"], 500, 25, word_split=True)
    return [{"text": chunk, "source": doc["source"]} for chunk in chunks]



def docx_2_html(filename="/data/data/game/itu/data/3GPP-Release18/Documents/22874-i20.docx"):

    with open(filename, "rb") as docx_file:
        result = mammoth.convert_to_html(docx_file)
        html = result.value  # The generated HTML
        messages = result.messages

    return messages,html



class DocEmbedding:
    """
    构建文档的embedding
    """
    def __init__(self,xinferenc_url="http://localhost:9998",model_uid = "bge-m3-embedding-0620",
                 folder_path = "/data/data/game/itu/data/3GPP-Release18/Documents/"):
        self.folder_path = folder_path
        self.client = Client(xinferenc_url)
        self.emb_model = self.client.get_model(model_uid)

    def get_doc_embedding_origin(self,save_folder= "/data/data/game/itu/data/3GPP-Release18/embedding_by_bgem3_chunk500/"):
        """
        telco 中提供的原始方法
        """

        storage_name = 'Documents.db'
        dataset_name = "Standard"

        storage = Storage(f'/data/data/game/itu/data/3GPP-Release18/{storage_name}')

        for filename in tqdm(os.listdir(self.folder_path)):
            print("------------------", filename)
            try:
                data_dict = storage.get_dict_by_id(dataset_name, filename)

                document_ds = chunk_doc(data_dict)
                for doc in document_ds:
                    """
                    doc {"text":"","source":'22101-i50.docx',"emb":[]}
                    """
                    max_retry = 5
                    retey = 0
                    while retey < max_retry:
                        try:
                            emb = self.emb_model.create_embedding(doc["text"])
                            doc["emb"] = emb["data"][0]["embedding"]
                            break

                        except Exception as e:
                            retry += 1
                            print("retry:", str(retry), doc)

                save_path = f"{save_folder}/{filename}.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(document_ds, f)

                print("embedding:", save_path)
            except Exception as e:
                print("error", e, filename)

    def get_doc_embedding(self, document_ds, filename,
                          save_folder="/data/data/game/itu/data/3GPP-Release18/embedding_by_bgem3_chunk500/"):
        """
        document_ds :[{"text":"","source":'22101-i50.docx',"emb":[]},{"text":"","source":'22101-i50.docx',"emb":[]},...]
        :return:
        """
        try:
            for doc in document_ds:
                """
                doc {"text":"","source":'22101-i50.docx',"emb":[]}
                """
                max_retry = 5
                retey = 0
                while retey < max_retry:
                    try:
                        emb = self.emb_model.create_embedding(doc["text"])
                        doc["emb"] = emb["data"][0]["embedding"]
                        break

                    except Exception as e:
                        retry += 1
                        print("retry:", str(retry), doc)

            save_path = f"{save_folder}/{filename}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(document_ds, f)
            print("embedding:", save_path)
        except Exception as e:
            print("error", e, filename)






def task1():
    """
    把文档转换成html文件
    """
    folder_path = "/data/data/game/itu/data/3GPP-Release18/Documents/"
    save_path = "/data/data/game/itu/data/3GPP-Release18/Documents_HTML/"
    for filename in tqdm(os.listdir(folder_path)):
        file_full_path = os.path.join(folder_path, filename)
        messages,html = docx_2_html(filename=file_full_path)

        save_full_path = save_path+filename+".html"
        with open(save_full_path,"w",encoding="utf-8") as f:
            f.write(html)

        print(save_full_path)



if __name__ == '__main__':

    task1()

    # docx_2_html()

























