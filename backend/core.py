import os
from typing import Any, Dict, List
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Pinecone
import pinecone


INDEX_NAME= "gst-demo-index"

pinecone.init(
    api_key='915407a4-96a7-4446-93ae-2d359253be26',
    environment='gcp-starter',
)
INDEX_NAME = INDEX_NAME


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa({"question": query, "chat_history": chat_history})



#print(run_llm("What is lang chain",""))