import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
groq_api_key=os.getenv("GROQ_API_KEY")

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)
vectorstoredb_local = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
llm=ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

load_dotenv() ## loading all the environment variable


##prompts
prompt=ChatPromptTemplate.from_template(
    """
Answer the following question based only on the provided context:
<context>
{context}
</context>
"""
)

document_chain=create_stuff_documents_chain(llm,prompt)
document_chain.invoke({
    "context":[Document(page_content="there are some important concepts in astrology")]
})
retriever=vectorstoredb_local.as_retriever()
retrieval_chain=create_retrieval_chain(retriever,document_chain)

## streamlit framework
st.title("Astrology chat bot")
input_text=st.text_input("What question you have in mind?")


if input_text:
    response = retrieval_chain.invoke({"input":input_text})
    st.write(response["answer"])


