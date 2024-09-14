# **utils.py**

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("No OpenAI API key found in Streamlit secrets or environment variables.")

@st.cache_resource
def load_chain():
    """
    Initialize and configure a conversational retrieval chain for answering user questions.
    :return: ConversationalRetrievalChain object
    """
    print("Current working directory:", os.getcwd())
    print("Contents of current directory:", os.listdir())

    if 'faiss_index' in os.listdir():
        print("Contents of faiss_index directory:", os.listdir("faiss_index"))
    else:
        print("faiss_index directory not found")

    try:
        # Load OpenAI embedding model
        embeddings = OpenAIEmbeddings()

        # Load OpenAI chat model
        llm = ChatOpenAI(temperature=0)

        # Load our local FAISS index as a retriever
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})

        # Create memory 'chat_history'
        memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history")

        # Create system prompt
        template = """
        You are an AI assistant for answering questions about the Blendle Employee Handbook.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer.
        If you don't know the answer, just say 'Sorry, I don't know ... 😔. 
        Don't try to make up an answer.
        If the question is not about the Blendle Employee Handbook, politely inform them that you are tuned to only answer questions about the Blendle Employee Handbook.

        {context}
        Question: {question}
        Helpful Answer:"""

        # Create the Conversational Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
            verbose=True
        )

        # Add system prompt to chain
        QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
        chain.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate(prompt=QA_CHAIN_PROMPT)

        return chain

    except Exception as e:
        print(f"An error occurred while loading the chain: {str(e)}")
        raise