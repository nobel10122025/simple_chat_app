import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain


embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
)
vectorstoredb_local = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
llm=Ollama(model="gemma2:2b")

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
st.title("Langchain Demo With Gemma Model")
input_text=st.text_input("What question you have in mind?")


if input_text:
    response = retrieval_chain.invoke({"input":input_text})
    st.write(response["answer"])


