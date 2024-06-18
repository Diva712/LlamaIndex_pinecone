import os
import pinecone

# os.environ["OPENAI_API_KEY"] = "put your keys here" 



from llama_index.core import GPTVectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

os.environ['PINECONE_API_KEY'] = "put your keys here" 
# environment is found next to API key in the console


# initialize connection to pinecone
pc = Pinecone(
        api_key= "put your keys here" 
    )

# create the index if it does not exist already
index_name = 'llama-index-pinecone'
# if index_name not in pc.list_indexes():
#     pinecone.create_index(
#         index_name,
#         dimension=1536,
#         metric='cosine'
#     )
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
# connect to the index
pinecone_index = pc.Index(index_name)
index = pc.Index(index_name)

namespace = '' # default namespace

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)



# setup the index/query process, ie the embedding model (and completion if used)
embed_model = OpenAIEmbedding(model='text-embedding-ada-002', embed_batch_size=100)

service_context = ServiceContext.from_defaults(embed_model=embed_model)

documents = SimpleDirectoryReader("data").load_data()

index = GPTVectorStoreIndex.from_documents(
    documents, storage_context=storage_context,
    service_context=service_context
)



query_engine = index.as_query_engine()

response = query_engine.query("who is Mr. Dursley ?")
print(response)
