import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class RAGEngine:
    def __init__(self, pdf_path):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vector_store = None
        self._ingest(pdf_path)

    def _ingest(self, pdf_path):
        # 1. Load
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        # 2. Split (Crucial for context window limits)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Store
        self.vector_store = Chroma.from_documents(documents=splits, embedding=self.embeddings)

    def query(self, question):
        if not self.vector_store:
            return "Database not initialized."
            
        # 4. Retrieve
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 5. Generate
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say you don't know. "
            "Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = chain.invoke({"input": question})
        return response["answer"]