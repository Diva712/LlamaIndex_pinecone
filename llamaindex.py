!pip install llama-index openai

import os
os.environ["OPENAI_API_KEY"] =  "API_KEY_OPEN_AI"


from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from openai import OpenAI
from IPython.display import Markdown, display

documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)
response = query_engine.query("who is Mr. Dursley ?")
display(Markdown(f"{response}"))

//MAke a data folder and create file.txt//
