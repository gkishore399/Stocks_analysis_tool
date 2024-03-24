
#Loading the documents
import os
import streamlit as st
import pickle
import time
import langchain
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


import os

os.getenv('OPENAI_API_KEY')

# Initialise LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500) 

# #get the data
urls=[
    'https://www.inshorts.com/en/news/bhutan-king-wangchuck-walked-pm-modi-till-his-plane-as-he-left-for-india-pics-go-viral-1711214652003'
]

loader=UnstructuredURLLoader(urls=urls)
data=loader.load()
#Split data to create chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)
print(docs)


# Create the embeddings of the chunks using openAIEmbeddings
embeddings = OpenAIEmbeddings()

# Pass the documents and embeddings inorder to create FAISS vector index
db = FAISS.from_documents(docs, embeddings)

print(db)




# # Storing vector index create in local
file_path="vector_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump(db, f)

# if os.path.exists(file_path):
#     with open(file_path, "rb") as f:
#         vectorIndex = pickle.load(f)

#Retrieve similar embeddings for a given question and call LLM to retrieve final answer
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=db.as_retriever())



query = "who meet modi?"


langchain.debug=True

chain({"question": query}, return_only_outputs=True)
