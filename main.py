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

from dotenv import load_dotenv
load_dotenv()

urls=[]
# Initialise LLM with required params
llm = OpenAI(temperature=0.9, max_tokens=500) 

st.header("Stocks Research Tool")
main_placeholder=st.empty()
st.sidebar.title("News Article links")
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    
process_url_clicked=st.sidebar.button("Process URL's")


if process_url_clicked:
    #Load data
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()
    main_placeholder.text("Data Loading started.....")
    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.'],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Data Splitter started.....")
    # As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
    docs = text_splitter.split_documents(data)
    # Create the embeddings of the chunks using openAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Pass the documents and embeddings inorder to create FAISS vector index
    db = FAISS.from_documents(docs, embeddings)
    
    # file_path="vector_index.pkl"
    # with open(file_path, "wb") as f:
    #     pickle.dump(db, f)
        
query=main_placeholder.text_input("QUESTION :")
if query:
    #Retrieve similar embeddings for a given question and call LLM to retrieve final answer
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=db.as_retriever())

    output=chain({"question": query}, return_only_outputs=True)

    st.header("Answer")
    st.write(output['answer'])
    
    
    sources=output.get('sources',"")
    if sources:
        st.subheader("SOURCES:")
        sources_list=sources.split("\n")
        for i in sources_list:
            st.write(i)
        
    
